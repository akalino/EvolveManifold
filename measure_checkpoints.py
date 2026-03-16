import os
import pickle
import re
import time
import resource

import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist
from tqdm import tqdm

from complex_persistence import compute_vr_diagrams
from metrics import (
    effective_rank,
    top_k_variance_fraction,
    mean_pairwise_distance,
    projection_residual,
    total_persistence_h1,
    betti_curve_from_diagram,
    betti_curve_area,
    betti_curve_peak,
    betti_curve_change
)
from ph_workflow import PHWorkflow
from projectors import (
    proj_to_k_plane,
    proj_to_sphere,
    proj_to_torus,
    proj_to_paraboloid,
)


CKPT_RE = re.compile(r"ckpt_epoch_(\d+)\.pkl$")


def load_checkpoint(_path):
    with open(_path, "rb") as f:
        return pickle.load(f)


def get_memory_mb():
    _kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return float(_kb) / 1024.0


def parse_k_from_model(_model_name, _default=8):
    m = re.search(r"k(\d+)", _model_name)
    if m is None:
        return _default
    return int(m.group(1))


def get_projection_fn(_mechanism, _model_name):
    k = parse_k_from_model(_model_name)

    if _mechanism == "linear_to_kplane":
        return lambda x: proj_to_k_plane(x, _k=k)

    if _mechanism == "nonlinear_to_kplane":
        return lambda x: proj_to_k_plane(x, _k=k)

    if _mechanism == "nonlinear_to_sphere":
        return lambda x: proj_to_sphere(x, _r=1.0)

    if _mechanism == "nonlinear_to_torus":
        return lambda x: proj_to_torus(x, _r_major=2.0, _r_minor=0.5)

    if _mechanism == "nonlinear_to_paraboloid":
        return lambda x: proj_to_paraboloid(x)

    return None


def checkpoint_paths_for_run(_run_dir):
    files = []
    for name in os.listdir(_run_dir):
        if CKPT_RE.match(name):
            files.append(os.path.join(_run_dir, name))
    files.sort()
    return files


def measure_run(_run_dir, _too_big=False, _ph_mode="full_vr"):
    parts = _run_dir.split(os.sep)
    experiment = parts[-3]
    mechanism = parts[-2]
    model = parts[-1]

    proj_fn = get_projection_fn(mechanism, model)
    ckpt_paths = checkpoint_paths_for_run(_run_dir)

    ph = PHWorkflow(_mode=_ph_mode,
                    _max_dim=1,
                    _sparse=0.2,
                    _too_big=_too_big,
                    _n_landmarks=250,
                    _seed=17,
                    _skip_every=2,
                    _knn_k=12,
                    _event_thresh=0.02,
                    _event_max_skip=5)
    betti_grid = None
    betti_ref_curve_h1 = None

    rows = []
    i = 0
    for ckpt_path in tqdm(ckpt_paths):
        payload = load_checkpoint(ckpt_path)
        x = payload["x"]
        epoch = payload["epoch"]

        t0 = time.perf_counter()
        dgms = ph.diagrams(x, epoch)
        ph_time_sec = time.perf_counter() - t0
        mem_mb = get_memory_mb()

        dgm1 = dgms[1]
        if betti_grid is None:
            if len(dgm1) > 0:
                finite_deaths = dgm1[:, 1][np.isfinite(dgm1[:, 1])]
                max_val = finite_deaths.max() if len(finite_deaths) > 0 else 1.0
            else:
                max_val = 1.0
            betti_grid = np.linspace(0.0, ph.max_edge_len, 200)
            betti_ref_curve_h1 = betti_curve_from_diagram(dgm1, betti_grid)

        row = {
            "experiment": experiment,
            "mechanism": mechanism,
            "model": model,
            "epoch": epoch,
            "effective_rank": effective_rank(x),
            "top_k_variance_fraction": top_k_variance_fraction(
                x,
                parse_k_from_model(model),
            ),
            #"mean_pairwise_distance": mean_pairwise_distance(x),
            "ph_mode": _ph_mode,
            "ph_recomputed": ph.last_recomputed,
            "ph_time_sec": ph_time_sec,
            "ph_mem": mem_mb,
            "total_persistence_h1": total_persistence_h1(dgms[1]),
            "betti_curve_area_h1": betti_curve_area(dgm1, betti_grid),
            "betti_curve_peak_h1": betti_curve_peak(dgm1, betti_grid),
            "betti_curve_change_h1": betti_curve_change(dgm1,
                                                        betti_ref_curve_h1, betti_grid)
        }

        if proj_fn is not None:
            row["projection_residual"] = projection_residual(x, proj_fn)
        else:
            row["projection_residual"] = None

        rows.append(row)

    return pd.DataFrame(rows)


def find_run_dirs(_root_dir="evolve_checkpoints"):
    run_dirs = []
    for dirpath, _, filenames in os.walk(_root_dir):
        if any(CKPT_RE.match(name) for name in filenames):
            run_dirs.append(dirpath)
    run_dirs.sort()
    return run_dirs


def main(_root_dir="evolve_checkpoints",
         _out_dir="metric_outputs",
         _ph_mode="full_vr"):
    os.makedirs(_out_dir, exist_ok=True)

    run_dirs = find_run_dirs(_root_dir)
    dfs = []

    for i, run_dir in enumerate(run_dirs, start=1):
        parts = run_dir.split(os.sep)
        mechanism = parts[-2]
        model = parts[-1]
        out_name = f"{_ph_mode}-{mechanism}__{model}.csv"
        out_path = os.path.join(_out_dir, out_name)

        if os.path.exists(out_path):
            print(f"[{i}/{len(run_dirs)}] skipping existing {out_path}")
            continue

        if "nonlinear_to_paraboloid" in run_dir and "spiked_gaussian" in run_dir and "mp1.0" in run_dir:
            print(f"retrying {run_dir} for now")
            print(f"[{i}/{len(run_dirs)}] measuring {run_dir} - {_ph_mode}")
            df_run = measure_run(run_dir, True, _ph_mode)
            dfs.append(df_run)
            df_run.to_csv(out_path, index=False)
        else:
            print(f"[{i}/{len(run_dirs)}] measuring {run_dir}, {_ph_mode}")
            df_run = measure_run(run_dir, False, _ph_mode)
            dfs.append(df_run)
            df_run.to_csv(out_path, index=False)

    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        df_all.to_csv(os.path.join(_out_dir,
                                   "{}_all_metrics.csv".format(_ph_mode)), index=False)


if __name__ == "__main__":
    #main(_ph_mode="full_vr")
    main(_ph_mode="landmark_vr")
    #main(_ph_mode="fixed_support_vr")
    #main(_ph_mode="fixed_knn_vr")
    #main(_ph_mode="event_driven")
