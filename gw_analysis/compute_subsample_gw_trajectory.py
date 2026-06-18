#!/usr/bin/env python3
"""
Compute Gromov-Wasserstein distances along one EvolveManifold parquet trajectory run.

Expected access-branch run layout:

  run_dir/
    manifest.json
    metadata.json
    checkpoints/
      ckpt_epoch_0000.parquet
      ckpt_epoch_0002.parquet
      ...

Example:

  python compute_subsample_gw_trajectory.py
    --run-dir /home/alex/evolve_local/evolve_collapse/evolve_checkpoints/collapse_ph/radial_collapse/clustered_gaussian_n1000_d50_k16__exponential__moderate__mp0.5__noise0.0__seed5 --output ~/evolve_local/evolve_collapse/gw_outputs/radial_run_gw_adjacent.parquet --mode adjacent --n-landmarks 256

  python compute_gw_parquet_trajectory.py \
    --run-dir "$EVOLVE_ROOT/evolve_checkpoints/collapse_ph/radial_collapse/<run_id>" \
    --output "$EVOLVE_ROOT/gw_outputs/radial_run_gw_adjacent.parquet" \
    --mode adjacent \
    --n-landmarks 256

  python compute_gw_parquet_trajectory.py \
  --run-dir "$EVOLVE_ROOT/evolve_checkpoints/collapse_ph/radial_collapse/<run_id>" \
  --output "$EVOLVE_ROOT/gw_outputs/<run_id>__gw_from_start.parquet" \
  --mode from_start \
  --n-landmarks 256 \
  --epsilon 0.005
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import ot


CKPT_PARQUET_RE = re.compile(r"ckpt_epoch_(\d+)\.parquet$")


def read_json_if_exists(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def epoch_from_checkpoint_path(path: str | Path) -> int:
    name = Path(path).name
    m = CKPT_PARQUET_RE.match(name)
    if m is None:
        return -1
    return int(m.group(1))


def checkpoint_paths_from_manifest(run_dir: str | Path) -> List[Path]:
    """
    Prefer manifest.json because the access branch treats it as the source of truth.
    Fall back to run_dir/checkpoints/ckpt_epoch_*.parquet if manifest paths are stale.
    """
    run_dir = Path(run_dir)
    manifest = read_json_if_exists(run_dir / "manifest.json")

    paths: List[Path] = []

    for item in manifest.get("checkpoints", []) or []:
        raw_path = item.get("path")
        if not raw_path:
            continue

        p = Path(raw_path)

        if not p.is_absolute():
            p = run_dir / p

        if p.exists() and CKPT_PARQUET_RE.match(p.name):
            paths.append(p)

    if paths:
        return sorted(paths, key=epoch_from_checkpoint_path)

    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.is_dir():
        paths.extend(ckpt_dir.glob("ckpt_epoch_*.parquet"))

    # Legacy-ish fallback in case parquet files are directly under run_dir.
    paths.extend(run_dir.glob("ckpt_epoch_*.parquet"))

    paths = sorted(set(paths), key=epoch_from_checkpoint_path)

    if not paths:
        raise FileNotFoundError(f"No parquet checkpoints found under run_dir={run_dir}")

    return paths


def load_checkpoint_parquet(path: str | Path) -> Tuple[np.ndarray, int]:
    """
    Return (x, epoch) from one parquet checkpoint.

    The access branch writes coordinate columns as dim_0000, dim_0001, ...
    """
    path = Path(path)
    epoch = epoch_from_checkpoint_path(path)

    if epoch < 0:
        raise ValueError(f"Could not parse epoch from checkpoint path: {path}")

    df = pd.read_parquet(path)

    dim_cols = [c for c in df.columns if str(c).startswith("dim_")]

    if dim_cols:
        dim_cols = sorted(dim_cols)
        x = df[dim_cols].to_numpy(dtype=float, copy=True)
    else:
        # Fallback for simple numeric-only parquet checkpoints.
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            raise ValueError(f"No numeric coordinate columns found in {path}")
        x = numeric.to_numpy(dtype=float, copy=True)

    if x.ndim != 2:
        raise ValueError(f"Expected 2D point cloud from {path}, got shape={x.shape}")

    return x, epoch


def parse_model_metadata(model: str) -> Dict[str, Any]:
    def find_int(pattern: str):
        m = re.search(pattern, model)
        return int(m.group(1)) if m else None

    def find_float(pattern: str):
        m = re.search(pattern, model)
        return float(m.group(1)) if m else None

    geom = None
    m = re.match(r"(.+)_n\d+_d\d+_k\d+__", model)
    if m:
        geom = m.group(1)

    return {
        "geometry": geom,
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

    model = (
        manifest.get("model")
        or manifest.get("run_id")
        or run_dir.name
    )

    parsed = parse_model_metadata(model)

    return {
        "run_id": manifest.get("run_id") or model,
        "run_dir": str(run_dir),
        "experiment": manifest.get("experiment") or metadata.get("experiment"),
        "mechanism": manifest.get("mechanism") or metadata.get("mechanism"),
        "model": model,
        "geometry": metadata.get("base_geometry") or metadata.get("geometry") or parsed["geometry"],
        "schedule": metadata.get("schedule"),
        "severity": metadata.get("severity"),
        "n": metadata.get("n", parsed["n"]),
        "d": metadata.get("d", parsed["d"]),
        "k": metadata.get("k", parsed["k"]),
        "seed": metadata.get("seed", parsed["seed"]),
        "mover_frac": metadata.get("mover_frac", parsed["mover_frac"]),
        "noise": metadata.get("noise", parsed["noise"]),
        "checkpoint_every": manifest.get("checkpoint_every"),
        "checkpoint_status": manifest.get("status"),
    }


def choose_landmarks(n: int, n_landmarks: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = min(int(n_landmarks), int(n))
    return np.sort(rng.choice(n, size=m, replace=False))


def positive_median(C: np.ndarray) -> float:
    positive = C[C > 0]
    if positive.size == 0:
        return 1.0
    scale = float(np.median(positive))
    if not np.isfinite(scale) or scale <= 0:
        return 1.0
    return scale


def make_cost_matrix(x: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    return cdist(x, x, metric=metric).astype(np.float64, copy=False)


def normalize_costs(
    costs: List[np.ndarray],
    normalize: str,
) -> Tuple[List[np.ndarray], float | None, List[float]]:
    """
    Important default: start_median.

    Do NOT default to per-snapshot median normalization for collapse work,
    because per-snapshot normalization can erase pure scale collapse.
    """
    medians = [positive_median(C) for C in costs]

    if normalize == "none":
        return costs, None, medians

    if normalize == "start_median":
        scale = medians[0]

    elif normalize == "global_median":
        scale = float(np.median(medians))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0

    elif normalize == "per_snapshot_median":
        normed = []
        for C, s in zip(costs, medians):
            if s <= 0:
                s = 1.0
            normed.append(C / s)
        return normed, None, medians

    else:
        raise ValueError(f"Unknown normalization mode: {normalize}")

    if scale <= 0:
        scale = 1.0

    return [C / scale for C in costs], scale, medians


def entropic_gw2(
    C1: np.ndarray,
    C2: np.ndarray,
    epsilon: float,
    max_iter: int,
    tol: float,
) -> float:
    p = np.ones(C1.shape[0], dtype=np.float64) / C1.shape[0]
    q = np.ones(C2.shape[0], dtype=np.float64) / C2.shape[0]

    val = ot.gromov.entropic_gromov_wasserstein2(
        C1,
        C2,
        p,
        q,
        loss_fun="square_loss",
        epsilon=epsilon,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
    )

    return float(val)


def build_pairs(num_checkpoints: int, mode: str) -> List[Tuple[int, int]]:
    if mode == "adjacent":
        return [(i, i + 1) for i in range(num_checkpoints - 1)]

    if mode == "from_start":
        return [(0, i) for i in range(1, num_checkpoints)]

    if mode == "full":
        return [
            (i, j)
            for i in range(num_checkpoints)
            for j in range(i + 1, num_checkpoints)
        ]

    raise ValueError(f"Unknown mode: {mode}")


def write_dataframe(df: pd.DataFrame, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    tmp = output.with_name(output.name + f".tmp.{os.getpid()}")

    if output.suffix == ".parquet":
        df.to_parquet(tmp, index=False)
    elif output.suffix == ".csv":
        df.to_csv(tmp, index=False)
    else:
        raise ValueError("Output must end in .parquet or .csv")

    os.replace(tmp, output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute GW distances along one parquet checkpoint trajectory."
    )

    parser.add_argument("--run-dir", required=True, help="One EvolveManifold run directory.")
    parser.add_argument("--output", required=True, help="Output .parquet or .csv path.")

    parser.add_argument(
        "--mode",
        choices=["adjacent", "from_start", "full"],
        default="adjacent",
        help="adjacent: GW(X_t, X_{t+1}); from_start: GW(X_0, X_t); full: all pairs.",
    )

    parser.add_argument("--n-landmarks", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)

    parser.add_argument(
        "--normalize",
        choices=["start_median", "global_median", "per_snapshot_median", "none"],
        default="start_median",
        help=(
            "start_median is recommended for collapse trajectories because it keeps "
            "scale changes relative to epoch 0. per_snapshot_median can hide pure shrinkage."
        ),
    )

    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--epsilon", type=float, default=5e-1)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-6)

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output = Path(args.output)

    meta = metadata_from_run_dir(run_dir)
    ckpt_paths = checkpoint_paths_from_manifest(run_dir)

    if len(ckpt_paths) < 2:
        raise ValueError(f"Need at least two checkpoints; found {len(ckpt_paths)}")

    print(f"[INFO] run_dir={run_dir}")
    print(f"[INFO] checkpoints={len(ckpt_paths)}")
    print(f"[INFO] mode={args.mode}")
    print(f"[INFO] n_landmarks={args.n_landmarks}")
    print(f"[INFO] normalize={args.normalize}")

    first_x, first_epoch = load_checkpoint_parquet(ckpt_paths[0])
    landmark_idx = choose_landmarks(first_x.shape[0], args.n_landmarks, args.seed)

    checkpoint_info: List[Dict[str, Any]] = []
    costs_raw: List[np.ndarray] = []

    for path in ckpt_paths:
        x, epoch = load_checkpoint_parquet(path)

        if x.shape[0] <= int(landmark_idx.max()):
            raise ValueError(
                f"Checkpoint {path} has n={x.shape[0]}, but landmark index "
                f"{int(landmark_idx.max())} was selected from the first checkpoint."
            )

        x_lm = x[landmark_idx]
        C = make_cost_matrix(x_lm, metric=args.metric)

        costs_raw.append(C)
        checkpoint_info.append(
            {
                "epoch": int(epoch),
                "path": str(path),
                "n_points": int(x.shape[0]),
                "ambient_dim": int(x.shape[1]),
            }
        )

    costs, common_scale, snapshot_medians = normalize_costs(
        costs_raw,
        normalize=args.normalize,
    )

    pairs = build_pairs(len(costs), args.mode)

    rows = []

    for pair_index, (i, j) in enumerate(pairs, start=1):
        source_epoch = checkpoint_info[i]["epoch"]
        target_epoch = checkpoint_info[j]["epoch"]

        print(
            f"[GW {pair_index}/{len(pairs)}] "
            f"epoch {source_epoch} -> {target_epoch}",
            flush=True,
        )

        t0 = time.perf_counter()

        gw2 = entropic_gw2(
            costs[i],
            costs[j],
            epsilon=args.epsilon,
            max_iter=args.max_iter,
            tol=args.tol,
        )

        elapsed = time.perf_counter() - t0

        rows.append(
            {
                **meta,
                "source_epoch": int(source_epoch),
                "target_epoch": int(target_epoch),
                "comparison_type": args.mode,
                "gw2": gw2,
                "gw_distance": float(np.sqrt(max(gw2, 0.0))),
                "gw_time_sec": float(elapsed),
                "source_checkpoint_path": checkpoint_info[i]["path"],
                "target_checkpoint_path": checkpoint_info[j]["path"],
                "source_snapshot_median_distance": float(snapshot_medians[i]),
                "target_snapshot_median_distance": float(snapshot_medians[j]),
                "normalization": args.normalize,
                "normalization_common_scale": common_scale,
                "n_landmarks": int(len(landmark_idx)),
                "landmark_seed": int(args.seed),
                "distance_metric": args.metric,
                "epsilon": float(args.epsilon),
                "max_iter": int(args.max_iter),
                "tol": float(args.tol),
            }
        )

    df = pd.DataFrame(rows)
    write_dataframe(df, output)

    meta_out = output.with_suffix(output.suffix + ".metadata.json")
    meta_payload = {
        "run_dir": str(run_dir),
        "output": str(output),
        "mode": args.mode,
        "num_checkpoints": len(ckpt_paths),
        "checkpoint_paths": [str(p) for p in ckpt_paths],
        "epochs": [info["epoch"] for info in checkpoint_info],
        "n_landmarks": int(len(landmark_idx)),
        "landmark_seed": int(args.seed),
        "landmark_indices": landmark_idx.tolist(),
        "normalization": args.normalize,
        "normalization_common_scale": common_scale,
        "snapshot_median_distances": snapshot_medians,
        "epsilon": args.epsilon,
        "max_iter": args.max_iter,
        "tol": args.tol,
    }

    meta_out.write_text(json.dumps(meta_payload, indent=2, sort_keys=True))

    print(f"[DONE] wrote {output}")
    print(f"[DONE] wrote {meta_out}")


if __name__ == "__main__":
    main()
