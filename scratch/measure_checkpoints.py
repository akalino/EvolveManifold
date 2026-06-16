import argparse
import os
import pickle
import re
import time
import resource

import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist
from tqdm import tqdm

from metrics import (
    effective_rank,
    top_k_variance_fraction,
    mean_pairwise_distance,
    projection_residual,
    total_persistence_h1,
    max_persistence_h1,
    top5_persistence_h1,
    betti_curve_from_diagram,
    betti_curve_area,
    betti_curve_peak,
    betti_curve_change,
)
from ph_workflow import PHWorkflow
from projectors import (
    proj_to_k_plane,
    proj_to_sphere,
    proj_to_torus,
    proj_to_paraboloid,
)

EXTERNAL_ROOT = "/media/alkal/WD_BLACK/evolve_collapse"
CHECKPOINT_ROOT = os.path.join(EXTERNAL_ROOT, "evolve_checkpoints")
METRIC_ROOT = os.path.join(EXTERNAL_ROOT, "../metric_outputs")
SUMMARY_ROOT = os.path.join(EXTERNAL_ROOT, "../metric_summaries")
ASSET_ROOT = os.path.join(EXTERNAL_ROOT, "summary_assets")

CKPT_RE = re.compile(r"ckpt_epoch_(\d+)\.pkl$")


def load_checkpoint(_path):
    with open(_path, "rb") as f:
        return pickle.load(f)


def get_memory_mb():
    _kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(_kb) / 1024.0


def safe_getattr(obj, name, default=None):
    return getattr(obj, name, default)


def pairwise_distance_summaries(x, max_points=1000, seed=17):
    """
    Returns lightweight pairwise-distance summaries.

    For large point clouds, this computes distances on a deterministic
    subsample so geometric baselines remain cheap enough for trajectory runs.
    """
    x = np.asarray(x)
    n = x.shape[0]

    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        x_eval = x[idx]
    else:
        x_eval = x

    if x_eval.shape[0] < 2:
        return {
            "mean_pairwise_distance": 0.0,
            "median_pairwise_distance": 0.0,
            "std_pairwise_distance": 0.0,
            "q10_pairwise_distance": 0.0,
            "q50_pairwise_distance": 0.0,
            "q90_pairwise_distance": 0.0,
            "pairwise_distance_num_points": int(x_eval.shape[0]),
        }

    dists = pdist(x_eval)

    return {
        "mean_pairwise_distance": float(np.mean(dists)),
        "median_pairwise_distance": float(np.median(dists)),
        "std_pairwise_distance": float(np.std(dists)),
        "q10_pairwise_distance": float(np.quantile(dists, 0.10)),
        "q50_pairwise_distance": float(np.quantile(dists, 0.50)),
        "q90_pairwise_distance": float(np.quantile(dists, 0.90)),
        "pairwise_distance_num_points": int(x_eval.shape[0]),
    }


def parse_k_from_model(_model_name, _default=8):
    m = re.search(r"k(\d+)", _model_name)
    if m is None:
        return _default
    return int(m.group(1))


def parse_model_metadata(model_name):
    """
    Parse metadata from model/run folder names.

    This is intentionally permissive. Missing fields are returned as None so
    downstream summaries do not crash if older runs have different names.
    """
    def find_float(pattern):
        m = re.search(pattern, model_name)
        return float(m.group(1)) if m else None

    def find_int(pattern):
        m = re.search(pattern, model_name)
        return int(m.group(1)) if m else None

    return {
        "n": find_int(r"n(\d+)"),
        "d": find_int(r"d(\d+)"),
        "k": find_int(r"k(\d+)"),
        "seed": find_int(r"seed(\d+)"),
        "mover_frac": find_float(r"mp([0-9.]+)"),
        "noise": find_float(r"noise([0-9.]+)"),
    }


def parse_experiment_metadata(experiment_name):
    """
    Parse geometry/schedule/severity when encoded in the experiment folder.
    Falls back to the raw experiment name when fields are unavailable.
    """
    out = {
        "geometry": experiment_name,
        "schedule": None,
        "severity": None,
    }

    # Examples this can handle if names include tokens like:
    # clustered_gaussian__linear__moderate
    # geom-clustered_gaussian_sched-linear_sev-strong
    m = re.search(r"geom-([^_]+(?:_[^_]+)*)", experiment_name)
    if m:
        out["geometry"] = m.group(1)

    m = re.search(r"sched-([A-Za-z0-9_]+)", experiment_name)
    if m:
        out["schedule"] = m.group(1)

    m = re.search(r"sev-([A-Za-z0-9_]+)", experiment_name)
    if m:
        out["severity"] = m.group(1)

    return out


def get_projection_fn(_mechanism, _model_name):
    k = parse_k_from_model(_model_name)
    print(f"[PROJ DIM] {k}")
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
    experiment_meta = parse_experiment_metadata(experiment)
    model_meta = parse_model_metadata(model)

    proj_fn = get_projection_fn(mechanism, model)
    ckpt_paths = checkpoint_paths_for_run(_run_dir)

    ph = PHWorkflow(_mode=_ph_mode,
                    _max_dim=1,
                    _sparse=0.2,
                    _too_big=_too_big,
                    _n_landmarks=500,
                    _seed=17,
                    _skip_every=2,
                    _knn_k=24,
                    _event_thresh=0.01,
                    _event_max_skip=5,
                    _force_every=5)
    betti_grid = None
    betti_ref_curve_h1 = None

    rows = []
    i = 0
    print("Beginning checkpoint measurement")
    for ckpt_path in tqdm(ckpt_paths):
        payload = load_checkpoint(ckpt_path)
        x = payload["x"]
        epoch = payload["epoch"]

        t0 = time.perf_counter()
        dgms = ph.diagrams(x, epoch)
        ph_time_sec = time.perf_counter() - t0
        # print(f"Diagram computation took {ph_time_sec} seconds")
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
        geom_summaries = pairwise_distance_summaries(
            x,
            max_points=1000,
            seed=model_meta.get("seed") or 17,
        )

        row = {
            "experiment": experiment,
            "geometry": experiment_meta["geometry"],
            "mechanism": mechanism,
            "model": model,
            "schedule": experiment_meta["schedule"],
            "severity": experiment_meta["severity"],
            "n": model_meta["n"],
            "d": model_meta["d"],
            "k": model_meta["k"],
            "seed": model_meta["seed"],
            "mover_frac": model_meta["mover_frac"],
            "noise": model_meta["noise"],
            "epoch": epoch,

            # Spectral metrics
            "effective_rank": effective_rank(x),
            "top_k_variance_fraction": top_k_variance_fraction(
                x,
                parse_k_from_model(model),
            ),

            # Geometric metrics
            **geom_summaries,

            # PH workflow metadata
            "ph_support_edges": safe_getattr(ph, "last_support_edges", None),
            "ph_support_refresh": safe_getattr(ph, "last_support_refresh", None),
            "ph_trigger": safe_getattr(ph, "last_trigger", None),
            "ph_mode": _ph_mode,
            "ph_recomputed": ph.last_recomputed,
            "ph_time_sec": ph_time_sec,
            "ph_mem": mem_mb,
            "ph_landmarks": ph.n_landmarks,
            "ph_event_score": ph.last_event_score,

            # Topological summaries
            "total_persistence_h1": total_persistence_h1(dgm1),
            "max_persistence_h1": max_persistence_h1(dgm1),
            "top5_persistence_h1": top5_persistence_h1(dgm1),
            "betti_curve_area_h1": betti_curve_area(dgm1, betti_grid),
            "betti_curve_peak_h1": betti_curve_peak(dgm1, betti_grid),
            "betti_curve_change_h1": betti_curve_change(
                dgm1,
                betti_ref_curve_h1,
                betti_grid,
            ),
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


def clean_name(s):
    return str(s).replace(os.sep, "_").replace(" ", "_")


def main(_root_dir="evolve_checkpoints",
         _out_dir="metric_outputs",
         _ph_mode="full_vr"):
    os.makedirs(_out_dir, exist_ok=True)

    run_dirs = find_run_dirs(_root_dir)
    dfs = []

    for i, run_dir in enumerate(run_dirs, start=1):
        parts = run_dir.rstrip(os.sep).split(os.sep)

        if len(parts) >= 3:
            experiment = parts[-3]
            mechanism = parts[-2]
            model = parts[-1]
        else:
            experiment = "unknown_experiment"
            mechanism = "unknown_mechanism"
            model = parts[-1]

        out_name = (
            f"{_ph_mode}__"
            f"{clean_name(experiment)}__"
            f"{clean_name(mechanism)}__"
            f"{clean_name(model)}.csv"
        )
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
    parser = argparse.ArgumentParser(
        description="Measure collapse metrics over checkpointed trajectories."
    )
    parser.add_argument(
        "--root-dir",
        default=CHECKPOINT_ROOT,
        help="Directory containing checkpointed trajectory runs.",
    )
    parser.add_argument(
        "--out-dir",
        default=METRIC_ROOT,
        help="Directory where metric CSVs will be written.",
    )
    parser.add_argument(
        "--ph-mode",
        default="online_landmark_dynamic_support",
        choices=[
            "full_vr",
            "landmark_vr",
            "skip_vr",
            "fixed_support_vr",
            "fixed_knn_vr",
            "event_driven",
            "online_landmark_event",
            "online_landmark_dynamic_support",
        ],
        help="Persistent homology workflow mode.",
    )
    args = parser.parse_args()

    main(
        _root_dir=args.root_dir,
        _out_dir=args.out_dir,
        _ph_mode=args.ph_mode,
    )
