"""
This is the measurement-side companion to the parquet checkpoint runner.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import time
import resource
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from tqdm import tqdm

from metrics import (
    effective_rank,
    top_k_variance_fraction,
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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

LOCAL_ROOT = os.path.expanduser("~/evolve_local/evolve_collapse")

EXTERNAL_ROOT = os.environ.get(
    "EVOLVE_COLLAPSE_ROOT",
    os.environ.get("EVOLVE_ROOT", LOCAL_ROOT),
)
CHECKPOINT_ROOT = os.path.join(EXTERNAL_ROOT, "evolve_checkpoints")
METRIC_ROOT = os.path.join(EXTERNAL_ROOT, "old/metric_outputs")
SUMMARY_ROOT = os.path.join(EXTERNAL_ROOT, "old/metric_summaries")
ASSET_ROOT = os.path.join(EXTERNAL_ROOT, "summary_assets")

OLD_MEDIA_PREFIX = "/media/alex/WD_BLACK"
CKPT_PARQUET_RE = re.compile(r"ckpt_epoch_(\d+)\.parquet$")
CKPT_PKL_RE = re.compile(r"ckpt_epoch_(\d+)\.pkl$")


def utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def atomic_write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def atomic_write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def guard_storage_path(path: str | Path, *, allow_old_media: bool = False) -> None:
    resolved = str(Path(path).expanduser())
    if resolved.startswith(OLD_MEDIA_PREFIX) and not allow_old_media:
        raise RuntimeError(
            f"Refusing to use old fragile path {resolved!r}. "
            "Use /mnt/wd_black/research/evolve_collapse or pass --allow-old-media."
        )


def require_writable_dir(path: str | Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    probe = path / f".write_probe_{os.getpid()}"
    probe.write_text("ok")
    probe.unlink()


def read_json_if_exists(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def load_checkpoint(path: str | Path) -> Tuple[np.ndarray, int]:
    """Return (x, epoch) from parquet or legacy pickle checkpoint."""
    path = Path(path)
    m = CKPT_PARQUET_RE.match(path.name)
    if m:
        epoch = int(m.group(1))
        df = pd.read_parquet(path)
        dim_cols = [c for c in df.columns if str(c).startswith("dim_")]
        if dim_cols:
            dim_cols = sorted(dim_cols)
            x = df[dim_cols].to_numpy(dtype=float, copy=True)
        else:
            x = df.to_numpy(dtype=float, copy=True)
        return x, epoch

    m = CKPT_PKL_RE.match(path.name)
    if m:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and "x" in payload:
            return np.asarray(payload["x"], dtype=float), int(payload.get("epoch", m.group(1)))
        return np.asarray(payload, dtype=float), int(m.group(1))

    raise ValueError(f"Unsupported checkpoint path: {path}")


def get_memory_mb() -> float:
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(kb) / 1024.0


def safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def pairwise_distance_summaries(x: np.ndarray, max_points: int = 1000, seed: int = 17) -> Dict[str, Any]:
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


def parse_k_from_model(model_name: str, default: int = 8) -> int:
    m = re.search(r"k(\d+)", model_name)
    if m is None:
        return default
    return int(m.group(1))


def parse_model_metadata(model_name: str) -> Dict[str, Any]:
    def find_float(pattern: str) -> Optional[float]:
        m = re.search(pattern, model_name)
        return float(m.group(1)) if m else None

    def find_int(pattern: str) -> Optional[int]:
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


def metadata_from_run_dir(run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    manifest = read_json_if_exists(run_dir / "manifest.json")
    metadata = read_json_if_exists(run_dir / "metadata.json")

    parts = run_dir.rstrip(os.sep).split(os.sep) if isinstance(run_dir, str) else str(run_dir).split(os.sep)
    experiment = manifest.get("experiment") or (parts[-3] if len(parts) >= 3 else "unknown_experiment")
    mechanism = manifest.get("mechanism") or metadata.get("mechanism") or (parts[-2] if len(parts) >= 2 else "unknown_mechanism")
    model = manifest.get("model") or manifest.get("run_id") or (parts[-1] if parts else "unknown_model")

    parsed_model = parse_model_metadata(model)

    # The checkpoint runner stores the most reliable experiment fields in metadata.json.
    out = {
        "run_id": manifest.get("run_id") or model,
        "run_dir": str(run_dir),
        "experiment": experiment,
        "mechanism": mechanism,
        "model": model,
        "geometry": metadata.get("base_geometry") or metadata.get("geometry") or parsed_geometry_from_model(model),
        "schedule": metadata.get("schedule"),
        "severity": metadata.get("severity"),
        "n": metadata.get("n", parsed_model.get("n")),
        "d": metadata.get("d", parsed_model.get("d")),
        "k": metadata.get("k", parsed_model.get("k")),
        "seed": metadata.get("seed", parsed_model.get("seed")),
        "mover_frac": metadata.get("mover_frac", parsed_model.get("mover_frac")),
        "noise": metadata.get("noise", parsed_model.get("noise")),
        "checkpoint_every": manifest.get("checkpoint_every"),
        "checkpoint_status": manifest.get("status"),
    }
    return out


def parsed_geometry_from_model(model: str) -> Optional[str]:
    # make_run_stem starts with base_geometry_n...; strip at _n<num>.
    m = re.match(r"(.+)_n\d+_d\d+_k\d+__", model)
    if m:
        return m.group(1)
    return None


def get_projection_fn(mechanism: str, model_name: str, k_value: Optional[int] = None):
    k = int(k_value or parse_k_from_model(model_name))
    if mechanism in {"linear_to_kplane", "nonlinear_to_kplane"}:
        return lambda x: proj_to_k_plane(x, _k=k)
    if mechanism == "nonlinear_to_sphere":
        return lambda x: proj_to_sphere(x, _r=1.0)
    if mechanism == "nonlinear_to_torus":
        return lambda x: proj_to_torus(x, _r_major=2.0, _r_minor=0.5)
    if mechanism == "nonlinear_to_paraboloid":
        return lambda x: proj_to_paraboloid(x)
    return None


def checkpoint_paths_from_manifest(run_dir: str | Path) -> List[Path]:
    run_dir = Path(run_dir)
    manifest = read_json_if_exists(run_dir / "manifest.json")
    paths: List[Path] = []

    for item in manifest.get("checkpoints", []) or []:
        raw_path = item.get("path")
        if not raw_path:
            continue
        p = Path(raw_path)
        if not p.is_absolute():
            p = run_dir / raw_path
        if p.exists():
            paths.append(p)

    if paths:
        return sorted(paths, key=lambda p: epoch_from_checkpoint_path(p))

    return checkpoint_paths_for_run(run_dir)


def epoch_from_checkpoint_path(path: str | Path) -> int:
    name = Path(path).name
    for regex in (CKPT_PARQUET_RE, CKPT_PKL_RE):
        m = regex.match(name)
        if m:
            return int(m.group(1))
    return -1


def checkpoint_paths_for_run(run_dir: str | Path) -> List[Path]:
    run_dir = Path(run_dir)
    candidates: List[Path] = []

    # New layout.
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.is_dir():
        candidates.extend(ckpt_dir.glob("ckpt_epoch_*.parquet"))
        candidates.extend(ckpt_dir.glob("ckpt_epoch_*.pkl"))

    # Legacy fallback.
    candidates.extend(run_dir.glob("ckpt_epoch_*.parquet"))
    candidates.extend(run_dir.glob("ckpt_epoch_*.pkl"))

    return sorted(set(candidates), key=lambda p: epoch_from_checkpoint_path(p))


def measure_run(run_dir: str | Path, too_big: bool = False, ph_mode: str = "full_vr") -> pd.DataFrame:
    run_dir = Path(run_dir)
    meta = metadata_from_run_dir(run_dir)
    proj_fn = get_projection_fn(meta["mechanism"], meta["model"], meta.get("k"))
    ckpt_paths = checkpoint_paths_from_manifest(run_dir)

    if not ckpt_paths:
        raise FileNotFoundError(f"No parquet or pickle checkpoints found for run: {run_dir}")

    ph = PHWorkflow(
        _mode=ph_mode,
        _max_dim=1,
        _sparse=0.2,
        _too_big=too_big,
        _n_landmarks=500,
        _seed=17,
        _skip_every=2,
        _knn_k=24,
        _event_thresh=0.01,
        _event_max_skip=5,
        _force_every=5,
    )
    betti_grid = None
    betti_ref_curve_h1 = None

    rows = []
    print(f"Beginning checkpoint measurement: {run_dir}")
    for ckpt_path in tqdm(ckpt_paths, desc=meta["model"]):
        x, epoch = load_checkpoint(ckpt_path)

        t0 = time.perf_counter()
        dgms = ph.diagrams(x, epoch)
        ph_time_sec = time.perf_counter() - t0
        mem_mb = get_memory_mb()

        dgm1 = dgms[1]
        if betti_grid is None:
            betti_grid = np.linspace(0.0, ph.max_edge_len, 200)
            betti_ref_curve_h1 = betti_curve_from_diagram(dgm1, betti_grid)

        geom_summaries = pairwise_distance_summaries(
            x,
            max_points=1000,
            seed=meta.get("seed") or 17,
        )

        k_for_metric = int(meta.get("k") or parse_k_from_model(meta["model"]))
        row = {
            **meta,
            "epoch": int(epoch),
            "checkpoint_path": str(ckpt_path),
            "checkpoint_format": ckpt_path.suffix.lstrip("."),
            "measured_at": utc_now_iso(),
            "effective_rank": effective_rank(x),
            "top_k_variance_fraction": top_k_variance_fraction(x, k_for_metric),
            **geom_summaries,
            "ph_support_edges": safe_getattr(ph, "last_support_edges", None),
            "ph_support_refresh": safe_getattr(ph, "last_support_refresh", None),
            "ph_trigger": safe_getattr(ph, "last_trigger", None),
            "ph_mode": ph_mode,
            "ph_recomputed": ph.last_recomputed,
            "ph_time_sec": ph_time_sec,
            "ph_mem": mem_mb,
            "ph_landmarks": ph.n_landmarks,
            "ph_event_score": ph.last_event_score,
            "total_persistence_h1": total_persistence_h1(dgm1),
            "max_persistence_h1": max_persistence_h1(dgm1),
            "top5_persistence_h1": top5_persistence_h1(dgm1),
            "betti_curve_area_h1": betti_curve_area(dgm1, betti_grid),
            "betti_curve_peak_h1": betti_curve_peak(dgm1, betti_grid),
            "betti_curve_change_h1": betti_curve_change(dgm1, betti_ref_curve_h1, betti_grid),
        }

        row["projection_residual"] = projection_residual(x, proj_fn) if proj_fn is not None else None
        rows.append(row)

    return pd.DataFrame(rows)


def clean_name(s: Any) -> str:
    return str(s).replace(os.sep, "_").replace(" ", "_")


def output_dir_for_run(run_dir: str | Path, out_root: str | Path, ph_mode: str) -> Path:
    meta = metadata_from_run_dir(run_dir)
    return (
        Path(out_root)
        / clean_name(ph_mode)
        / clean_name(meta["experiment"])
        / clean_name(meta["mechanism"])
        / clean_name(meta["model"])
    )


def output_paths_for_run(run_dir: str | Path, out_root: str | Path, ph_mode: str) -> Dict[str, Path]:
    out_dir = output_dir_for_run(run_dir, out_root, ph_mode)
    return {
        "out_dir": out_dir,
        "metrics_parquet": out_dir / "metrics.parquet",
        "metrics_csv": out_dir / "metrics.csv",
        "manifest": out_dir / "measurement_manifest.json",
    }


def should_use_too_big(run_dir: str | Path) -> bool:
    s = str(run_dir)
    return "nonlinear_to_paraboloid" in s and "spiked_gaussian" in s and "mp1.0" in s



def canonical_regime_pair(meta: Dict[str, Any]) -> bool:
    """Small full-VR audit/calibration panel."""
    return (
        (meta.get("geometry") == "spiked_gaussian" and meta.get("mechanism") == "linear_to_kplane")
        or (meta.get("geometry") == "isotropic" and meta.get("mechanism") == "radial_collapse")
        or (meta.get("geometry") == "clustered_gaussian" and meta.get("mechanism") == "cluster_tightening")
        or (meta.get("geometry") == "clustered_gaussian" and meta.get("mechanism") == "cluster_merging")
        or (meta.get("geometry") == "torus" and meta.get("mechanism") == "hole_fill")
    )


def in_tranche(meta: Dict[str, Any], tranche: str) -> bool:
    """Named measurement gates for controlled benchmark scaling."""
    if tranche in {"all", "", None}:
        return True

    geometry = meta.get("geometry")
    mechanism = meta.get("mechanism")
    schedule = meta.get("schedule")
    severity = meta.get("severity")
    n = meta.get("n")
    d = meta.get("d")
    seed = meta.get("seed")
    mover_frac = meta.get("mover_frac")
    noise = meta.get("noise")

    main_geometries = {
        "clustered_gaussian",
        "spiked_gaussian",
        "torus",
        "isotropic",
        "sphere",
        "swiss",
    }
    main_mechanisms = {
        "linear_to_kplane",
        "radial_collapse",
        "cluster_tightening",
        "cluster_merging",
        "hole_fill",
    }
    core_seeds = {5, 17, 26, 31, 37, 51, 123, 821, 1111, 1823}
    audit_seeds = {5, 17, 26}
    core_schedules = {"linear", "sigmoid"}
    core_severities = {"moderate", "strong"}
    core_mover_fracs = {0.25, 1.0}

    if tranche == "canonical":
        return (
            canonical_regime_pair(meta)
            and n == 1000
            and d == 50
            and schedule == "linear"
            and severity == "moderate"
            and mover_frac == 1.0
            and noise == 0.0
            and seed in core_seeds
        )

    if tranche == "primary_d50":
        return (
            geometry in main_geometries
            and mechanism in main_mechanisms
            and n == 1000
            and d == 50
            and schedule in core_schedules
            and severity in core_severities
            and mover_frac in core_mover_fracs
            and noise == 0.0
            and seed in core_seeds
        )

    if tranche == "primary_d100":
        return (
            geometry in main_geometries
            and mechanism in main_mechanisms
            and n == 1000
            and d == 100
            and schedule in core_schedules
            and severity in core_severities
            and mover_frac in core_mover_fracs
            and noise == 0.0
            and seed in core_seeds
        )

    if tranche == "noise_robustness":
        return (
            geometry in main_geometries
            and mechanism in main_mechanisms
            and n == 1000
            and d == 50
            and schedule == "linear"
            and severity in core_severities
            and mover_frac in core_mover_fracs
            and noise in {0.01, 0.05}
            and seed in audit_seeds
        )

    if tranche == "n_scaling":
        return (
            geometry in main_geometries
            and mechanism in main_mechanisms
            and n in {500, 2000}
            and d == 50
            and schedule == "linear"
            and severity == "moderate"
            and mover_frac == 1.0
            and noise == 0.0
            and seed in audit_seeds
        )

    if tranche == "ph_audit":
        return (
            canonical_regime_pair(meta)
            and n == 1000
            and d in {50, 100}
            and schedule == "linear"
            and severity == "moderate"
            and mover_frac == 1.0
            and noise == 0.0
            and seed in audit_seeds
        )

    raise ValueError(
        f"Unknown tranche={tranche!r}. Expected one of: "
        "all, canonical, primary_d50, primary_d100, noise_robustness, "
        "n_scaling, ph_audit."
    )


def filter_run_dirs_by_tranche(run_dirs: Iterable[Path], tranche: str) -> List[Path]:
    kept: List[Path] = []
    for run_dir in run_dirs:
        meta = metadata_from_run_dir(run_dir)
        if in_tranche(meta, tranche):
            kept.append(Path(run_dir))
    return sorted(kept, key=lambda p: str(p))


def find_run_dirs(root_dir: str | Path, include_incomplete: bool = False) -> List[Path]:
    root_dir = Path(root_dir)
    run_dirs: List[Path] = []

    # Prefer new manifest-driven discovery.
    for manifest_path in root_dir.rglob("manifest.json"):
        run_dir = manifest_path.parent
        manifest = read_json_if_exists(manifest_path)
        status = manifest.get("status", "unknown")
        if status == "completed" or include_incomplete:
            if checkpoint_paths_from_manifest(run_dir):
                run_dirs.append(run_dir)

    # Legacy fallback: directories containing old checkpoint files and no manifest.
    for dirpath, _, filenames in os.walk(root_dir):
        if "manifest.json" in filenames:
            continue
        if any(CKPT_PKL_RE.match(name) or CKPT_PARQUET_RE.match(name) for name in filenames):
            run_dirs.append(Path(dirpath))

    return sorted(set(run_dirs), key=lambda p: str(p))


def measure_one_run_task(args: Tuple[str, str, str, bool, bool]) -> Dict[str, Any]:
    run_dir_s, out_root_s, ph_mode, overwrite, write_csv = args
    run_dir = Path(run_dir_s)
    paths = output_paths_for_run(run_dir, out_root_s, ph_mode)
    measurement_manifest = read_json_if_exists(paths["manifest"])

    if (
        not overwrite
        and paths["metrics_parquet"].exists()
        and measurement_manifest.get("status") == "completed"
    ):
        print(f"Skipped completed measurement: {paths['metrics_parquet']}")
        return {
            "run_dir": str(run_dir),
            "out_dir": str(paths["out_dir"]),
            "metrics_parquet": str(paths["metrics_parquet"]),
            "metrics_csv": str(paths["metrics_csv"]),
            "status": "skipped_existing",
            "error": None,
            "updated_at": utc_now_iso(),
        }

    started_at = utc_now_iso()
    atomic_write_json(paths["manifest"], {
        "status": "running",
        "run_dir": str(run_dir),
        "ph_mode": ph_mode,
        "started_at": started_at,
        "updated_at": started_at,
        "metrics_parquet": str(paths["metrics_parquet"]),
        "metrics_csv": str(paths["metrics_csv"]),
    })

    try:
        print(f"Starting measurement: {paths['metrics_parquet']}")
        too_big = should_use_too_big(run_dir)
        df_run = measure_run(run_dir, too_big=too_big, ph_mode=ph_mode)
        atomic_write_parquet(df_run, paths["metrics_parquet"])
        if write_csv:
            atomic_write_csv(df_run, paths["metrics_csv"])
        completed_at = utc_now_iso()
        atomic_write_json(paths["manifest"], {
            "status": "completed",
            "run_dir": str(run_dir),
            "out_dir": str(paths["out_dir"]),
            "ph_mode": ph_mode,
            "started_at": started_at,
            "completed_at": completed_at,
            "updated_at": completed_at,
            "metrics_parquet": str(paths["metrics_parquet"]),
            "metrics_csv": str(paths["metrics_csv"]) if write_csv else None,
            "num_rows": int(len(df_run)),
            "columns": list(df_run.columns),
        })
        print(f"Done measurement: {paths['metrics_parquet']}")
        return {
            "run_dir": str(run_dir),
            "out_dir": str(paths["out_dir"]),
            "metrics_parquet": str(paths["metrics_parquet"]),
            "metrics_csv": str(paths["metrics_csv"]) if write_csv else None,
            "status": "completed",
            "error": None,
            "updated_at": completed_at,
        }
    except Exception as exc:
        failed_at = utc_now_iso()
        atomic_write_json(paths["manifest"], {
            "status": "failed",
            "run_dir": str(run_dir),
            "out_dir": str(paths["out_dir"]),
            "ph_mode": ph_mode,
            "started_at": started_at,
            "updated_at": failed_at,
            "metrics_parquet": str(paths["metrics_parquet"]),
            "metrics_csv": str(paths["metrics_csv"]) if write_csv else None,
            "error": repr(exc),
        })
        print(f"Failed measurement {paths['metrics_parquet']}: {exc}")
        return {
            "run_dir": str(run_dir),
            "out_dir": str(paths["out_dir"]),
            "metrics_parquet": str(paths["metrics_parquet"]),
            "metrics_csv": str(paths["metrics_csv"]) if write_csv else None,
            "status": "failed",
            "error": repr(exc),
            "updated_at": failed_at,
        }


def combine_metric_outputs(results_df: pd.DataFrame, out_root: str | Path, ph_mode: str, write_csv: bool) -> None:
    parquet_paths = [p for p in results_df.get("metrics_parquet", pd.Series(dtype=str)).dropna().tolist() if Path(p).exists()]
    frames = []
    for path in tqdm(parquet_paths, desc="combining per-run parquet metrics"):
        try:
            df = pd.read_parquet(path)
            if len(df) > 0:
                frames.append(df)
        except Exception as exc:
            print(f"[WARN] could not read {path}: {exc}")

    if not frames:
        print("[WARN] no metric frames found to combine")
        return

    df_all = pd.concat(frames, ignore_index=True)
    combined_dir = Path(out_root) / clean_name(ph_mode) / "_combined"
    all_parquet = combined_dir / "all_metrics.parquet"
    atomic_write_parquet(df_all, all_parquet)
    print(f"[DONE] wrote combined metrics: {all_parquet}")
    if write_csv:
        all_csv = combined_dir / "all_metrics.csv"
        atomic_write_csv(df_all, all_csv)
        print(f"[DONE] wrote combined CSV: {all_csv}")
    print(f"[DONE] rows={len(df_all)}, columns={len(df_all.columns)}")


def main(
    root_dir: str = CHECKPOINT_ROOT,
    out_dir: str = METRIC_ROOT,
    ph_mode: str = "full_vr",
    workers: int = 1,
    include_incomplete: bool = False,
    overwrite: bool = False,
    write_csv: bool = True,
    allow_old_media: bool = False,
    tranche: str = "all",
) -> None:
    guard_storage_path(root_dir, allow_old_media=allow_old_media)
    guard_storage_path(out_dir, allow_old_media=allow_old_media)
    require_writable_dir(out_dir)

    discovered_run_dirs = find_run_dirs(root_dir, include_incomplete=include_incomplete)
    run_dirs = filter_run_dirs_by_tranche(discovered_run_dirs, tranche)
    print(f"[INFO] root_dir={root_dir}")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] discovered {len(discovered_run_dirs)} run directories")
    print(f"[INFO] tranche={tranche}")
    print(f"[INFO] selected {len(run_dirs)} run directories")
    print(f"[INFO] ph_mode={ph_mode}")
    print(f"[INFO] workers={workers}")

    tasks = [(str(run_dir), out_dir, ph_mode, overwrite, write_csv) for run_dir in run_dirs]
    results: List[Dict[str, Any]] = []

    if workers <= 1:
        for i, task in enumerate(tasks, start=1):
            print(f"[RUN {i}/{len(tasks)}] {task[0]}", flush=True)
            results.append(measure_one_run_task(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = []
            for i, task in enumerate(tasks, start=1):
                print(f"[PARENT] submit {i}/{len(tasks)}: {task[0]}", flush=True)
                futures.append(ex.submit(measure_one_run_task, task))
            for j, fut in enumerate(as_completed(futures), start=1):
                print(f"[PARENT] completed future {j}/{len(futures)}", flush=True)
                results.append(fut.result())

    results_df = pd.DataFrame(results)
    status_dir = Path(out_dir) / clean_name(ph_mode) / "_status"
    status_path = status_dir / "measurement_status.csv"
    atomic_write_csv(results_df, status_path)
    atomic_write_json(status_dir / "measurement_status.json", {
        "status": "completed",
        "updated_at": utc_now_iso(),
        "root_dir": root_dir,
        "out_dir": out_dir,
        "ph_mode": ph_mode,
        "tranche": tranche,
        "workers": workers,
        "num_runs": int(len(results_df)),
        "status_counts": results_df["status"].value_counts().to_dict() if len(results_df) else {},
    })
    print(f"[DONE] wrote status: {status_path}")

    if len(results_df) and (results_df["status"] == "failed").any():
        failed_path = status_dir / "measurement_failed.csv"
        atomic_write_csv(results_df[results_df["status"] == "failed"], failed_path)
        print(f"[WARN] wrote failed measurements: {failed_path}")

    combine_metric_outputs(results_df, out_dir, ph_mode, write_csv=write_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure metrics over parquet checkpointed trajectories.")
    parser.add_argument("--root-dir", default=CHECKPOINT_ROOT, help="Root directory containing checkpointed trajectory runs.")
    parser.add_argument("--out-dir", default=METRIC_ROOT, help="Root directory where metric outputs will be written.")
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
    parser.add_argument(
        "--tranche",
        default="all",
        choices=[
            "all",
            "canonical",
            "primary_d50",
            "primary_d100",
            "noise_robustness",
            "n_scaling",
            "ph_audit",
        ],
        help="Named gate selecting a controlled benchmark slice to measure.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes. Use 1 first.")
    parser.add_argument("--include-incomplete", action="store_true", help="Measure runs even if their checkpoint manifest is not completed.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite completed metric outputs for matching runs.")
    parser.add_argument("--no-csv", action="store_true", help="Only write parquet outputs, not CSV copies.")
    parser.add_argument("--allow-old-media", action="store_true", help="Allow paths under /media/alex/WD_BLACK for legacy recovery only.")
    args = parser.parse_args()

    main(
        root_dir=args.root_dir,
        out_dir=args.out_dir,
        ph_mode=args.ph_mode,
        workers=args.workers,
        include_incomplete=args.include_incomplete,
        overwrite=args.overwrite,
        write_csv=not args.no_csv,
        allow_old_media=args.allow_old_media,
        tranche=args.tranche,
    )
