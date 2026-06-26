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

TODO:
  --ot-method entropic_gw *
  --ot-method gw *
  --ot-method fused_gw *
  --ot-method sliced_wasserstein
  --ot-method wasserstein_identity

TODO:
  --landmark-method random
  --landmark-method fps
  --landmark-method stratified_label
  --landmark-method stratified_cluster

  Outputs:
    landmark_method
    landmark_seed
    landmark_coverage_mean
    landmark_coverage_q95
    landmark_coverage_max

"""

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


def read_json_if_exists(path):
    """
    Read a JSON file if it exists.

    :param path: JSON path.
    :return: Parsed dictionary, or an empty dictionary on failure.
    """
    path = Path(path)

    if not path.exists():
        return {}

def read_json_if_exists(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def epoch_from_checkpoint_path(path):
    """
    Parse an epoch integer from a checkpoint filename.

    :param path: Checkpoint path.
    :return: Parsed epoch, or -1 if parsing fails.
    """
    name = Path(path).name
    match = CKPT_PARQUET_RE.match(name)

    if match is None:
        return -1

    return int(match.group(1))


def checkpoint_paths_from_manifest(run_dir):
    """
    Find checkpoint parquet files for a run.

    The manifest is preferred because the access branch treats it as the source
    of truth. If manifest paths are missing or stale, this falls back to
    ``run_dir/checkpoints/ckpt_epoch_*.parquet`` and then direct checkpoint
    files under ``run_dir``.

    :param run_dir: Run directory.
    :return: Sorted list of checkpoint paths.
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

    paths = []
    paths: List[Path] = []

    for item in manifest.get("checkpoints", []) or []:
        raw_path = item.get("path")
        if not raw_path:
            continue

        path = Path(raw_path)
        if not path.is_absolute():
            path = run_dir / path

        if path.exists() and CKPT_PARQUET_RE.match(path.name):
            paths.append(path)
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

    paths.extend(run_dir.glob("ckpt_epoch_*.parquet"))
    if ckpt_dir.is_dir():
        paths.extend(ckpt_dir.glob("ckpt_epoch_*.parquet"))

    # Legacy-ish fallback in case parquet files are directly under run_dir.
    paths.extend(run_dir.glob("ckpt_epoch_*.parquet"))

    paths = sorted(set(paths), key=epoch_from_checkpoint_path)

    if not paths:
        raise FileNotFoundError(f"No parquet checkpoints found under run_dir={run_dir}")

    return paths


def coordinate_columns(df):
    """
    Find coordinate columns in a checkpoint dataframe.

    Columns named ``dim_*`` are preferred. If unavailable, all numeric columns
    are used as a fallback.

    :param df: Checkpoint dataframe.
    :return: List of coordinate columns.
    """
    dim_cols = sorted([col for col in df.columns if str(col).startswith("dim_")])

    if dim_cols:
        return dim_cols

    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        return []

    return list(numeric.columns)


def load_checkpoint_dataframe(path):
    """
    Load a parquet checkpoint dataframe and its epoch.

    :param path: Checkpoint path.
    :return: Pair ``(dataframe, epoch)``.
def load_checkpoint_parquet(path: str | Path) -> Tuple[np.ndarray, int]:
    """
    Return (x, epoch) from one parquet checkpoint.

    The access branch writes coordinate columns as dim_0000, dim_0001, ...
    """
    path = Path(path)
    epoch = epoch_from_checkpoint_path(path)

    if epoch < 0:
        raise ValueError(f"Could not parse epoch from checkpoint path: {path}")

    return pd.read_parquet(path), epoch


def dataframe_to_point_cloud(df, path=None):
    """
    Convert a checkpoint dataframe to a point-cloud array.

    :param df: Checkpoint dataframe.
    :param path: Optional path used only for clearer error messages.
    :return: Point cloud as a numpy array.
    """
    cols = coordinate_columns(df)

    if not cols:
        if path is None:
            raise ValueError("No numeric coordinate columns found")
        raise ValueError(f"No numeric coordinate columns found in {path}")

    x = df[cols].to_numpy(dtype=float, copy=True)

    if x.ndim != 2:
        if path is None:
            raise ValueError(f"Expected 2D point cloud, got shape={x.shape}")
        raise ValueError(f"Expected 2D point cloud from {path}, got shape={x.shape}")

    return x


def load_checkpoint_parquet(path):
    """
    Load a checkpoint point cloud and epoch from parquet.

    :param path: Checkpoint path.
    :return: Pair ``(x, epoch)``.
    """
    df, epoch = load_checkpoint_dataframe(path)
    return dataframe_to_point_cloud(df, path=path), epoch


def parse_model_metadata(model):
    """
    Parse useful metadata from a run model string.

    :param model: Model or run identifier.
    :return: Parsed metadata dictionary.
    """
    def find_int(pattern):
        match = re.search(pattern, model)
        if match:
            return int(match.group(1))
        return None

    def find_float(pattern):
        match = re.search(pattern, model)
        if match:
            return float(match.group(1))
        return None

    geom = None
    match = re.match(r"(.+)_n\d+_d\d+_k\d+__", model)

    if match:
        geom = match.group(1)
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


def metadata_from_run_dir(run_dir):
    """
    Gather run metadata from manifest, metadata file, and run name.

    :param run_dir: Run directory.
    :return: Metadata dictionary.
    """
def metadata_from_run_dir(run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    manifest = read_json_if_exists(run_dir / "manifest.json")
    metadata = read_json_if_exists(run_dir / "metadata.json")

    model = manifest.get("model") or manifest.get("run_id") or run_dir.name
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


def random_landmarks(n, n_landmarks, seed):
    """
    Choose random landmark indices.

    :param n: Number of points.
    :param n_landmarks: Requested number of landmarks.
    :param seed: Random seed.
    :return: Sorted landmark indices.
    """
def choose_landmarks(n: int, n_landmarks: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = min(int(n_landmarks), int(n))
    return np.sort(rng.choice(n, size=m, replace=False))


def fps_landmarks(x, n_landmarks, seed):
    """
    Choose landmarks by greedy farthest-point sampling.

    :param x: Initial point cloud.
    :param n_landmarks: Requested number of landmarks.
    :param seed: Random seed for the first landmark.
    :return: Sorted landmark indices.
    """
    n = x.shape[0]
    m = min(int(n_landmarks), int(n))

    rng = np.random.default_rng(seed)
    idx = np.empty(m, dtype=int)
    idx[0] = int(rng.integers(0, n))

    d2 = np.sum((x - x[idx[0]]) ** 2, axis=1)

    for k in range(1, m):
        idx[k] = int(np.argmax(d2))
        new_d2 = np.sum((x - x[idx[k]]) ** 2, axis=1)
        d2 = np.minimum(d2, new_d2)

    return np.sort(idx)


def stratified_label_landmarks(df, n_landmarks, seed, label_col):
    """
    Choose landmarks by proportional label-stratified sampling.

    :param df: Initial checkpoint dataframe.
    :param n_landmarks: Requested number of landmarks.
    :param seed: Random seed.
    :param label_col: Label column name.
    :return: Sorted landmark indices.
    """
    if label_col not in df.columns:
        raise ValueError(f"Missing label column for stratified sampling: {label_col}")

    rng = np.random.default_rng(seed)
    n = len(df)
    m = min(int(n_landmarks), int(n))
    labels = df[label_col].to_numpy()
    counts = pd.Series(labels).value_counts(dropna=False)

    selected = []

    for label, count in counts.items():
        frac = float(count) / float(n)
        take = max(1, int(round(frac * m)))
        label_idx = np.flatnonzero(labels == label)
        take = min(take, len(label_idx))
        chosen = rng.choice(label_idx, size=take, replace=False)
        selected.extend(chosen.tolist())

    selected = np.array(sorted(set(selected)), dtype=int)

    if len(selected) > m:
        selected = np.sort(rng.choice(selected, size=m, replace=False))

    if len(selected) < m:
        missing = m - len(selected)
        pool = np.setdiff1d(np.arange(n), selected)
        extra = rng.choice(pool, size=missing, replace=False)
        selected = np.sort(np.concatenate([selected, extra]))

    return selected


def cluster_landmarks(x, n_landmarks, seed):
    """
    Choose a simple cluster-spread landmark set.

    This starter implementation uses farthest-point sampling as a deterministic
    coverage proxy. It is intentionally kept as a separate method name so it can
    later be replaced by k-means or medoid-based stratification without changing
    downstream CLI usage.

    :param x: Initial point cloud.
    :param n_landmarks: Requested number of landmarks.
    :param seed: Random seed.
    :return: Sorted landmark indices.
    """
    return fps_landmarks(x, n_landmarks, seed)


def choose_landmarks(x, n_landmarks, seed, method, df=None, label_col="label"):
    """
    Choose landmark indices using a named strategy.

    :param x: Initial point cloud.
    :param n_landmarks: Requested number of landmarks.
    :param seed: Random seed.
    :param method: Landmark method name.
    :param df: Optional initial checkpoint dataframe for label strategies.
    :param label_col: Label column for stratified label sampling.
    :return: Sorted landmark indices.
    """
    if method == "random":
        return random_landmarks(x.shape[0], n_landmarks, seed)

    if method == "fps":
        return fps_landmarks(x, n_landmarks, seed)

    if method == "stratified_label":
        if df is None:
            raise ValueError("stratified_label requires the first checkpoint dataframe")
        return stratified_label_landmarks(df, n_landmarks, seed, label_col)

    if method == "stratified_cluster":
        return cluster_landmarks(x, n_landmarks, seed)

    raise ValueError(f"Unknown landmark method: {method}")


def landmark_coverage_stats(x, landmark_idx):
    """
    Compute coverage diagnostics for a landmark subset.

    :param x: Full point cloud.
    :param landmark_idx: Landmark indices.
    :return: Coverage statistics dictionary.
    """
    landmarks = x[landmark_idx]
    dists = cdist(x, landmarks, metric="euclidean")
    nearest = np.min(dists, axis=1)
    quantiles = np.quantile(nearest, [0.5, 0.9, 0.95, 0.99, 1.0])

    return {
        "landmark_coverage_mean": float(np.mean(nearest)),
        "landmark_coverage_std": float(np.std(nearest)),
        "landmark_coverage_median": float(quantiles[0]),
        "landmark_coverage_q90": float(quantiles[1]),
        "landmark_coverage_q95": float(quantiles[2]),
        "landmark_coverage_q99": float(quantiles[3]),
        "landmark_coverage_max": float(quantiles[4]),
    }


def positive_median(cost):
    """
    Compute the positive median of a cost matrix.

    :param cost: Cost matrix.
    :return: Positive median with safe fallback.
    """
    positive = cost[cost > 0]

    if positive.size == 0:
        return 1.0

    scale = float(np.median(positive))

    if not np.isfinite(scale) or scale <= 0:
        return 1.0

    return scale


def make_cost_matrix(x, metric="euclidean"):
    """
    Build an internal cost matrix from a point cloud.

    :param x: Point cloud.
    :param metric: Distance metric passed to ``scipy.spatial.distance.cdist``.
    :return: Pairwise cost matrix.
    """
    return cdist(x, x, metric=metric).astype(np.float64, copy=False)


def normalize_costs(costs, normalize):
    """
    Normalize checkpoint cost matrices.

    ``start_median`` is the recommended default for collapse work because it
    preserves scale changes relative to the initial checkpoint. In contrast,
    ``per_snapshot_median`` can erase pure scale collapse.

    :param costs: List of raw cost matrices.
    :param normalize: Normalization mode.
    :return: Tuple ``(normalized_costs, common_scale, snapshot_medians)``.
    """
    medians = [positive_median(cost) for cost in costs]
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
        for cost, scale in zip(costs, medians):
            if scale <= 0:
                scale = 1.0
            normed.append(cost / scale)
        for C, s in zip(costs, medians):
            if s <= 0:
                s = 1.0
            normed.append(C / s)
        return normed, None, medians

    else:
        raise ValueError(f"Unknown normalization mode: {normalize}")

    if scale <= 0:
        scale = 1.0

    return [cost / scale for cost in costs], scale, medians


def normalize_cross_costs(cross_costs, normalize, common_scale, source_medians, target_medians):
    """
    Normalize cross-cost matrices for fused GW.

    :param cross_costs: List of cross-cost matrices keyed by pair order.
    :param normalize: Normalization mode.
    :param common_scale: Common scale from internal-cost normalization.
    :param source_medians: Source snapshot medians for each pair.
    :param target_medians: Target snapshot medians for each pair.
    :return: Normalized cross-cost matrices.
    """
    if normalize == "none":
        return cross_costs

    if normalize in ["start_median", "global_median"]:
        scale = common_scale
        if scale is None or scale <= 0:
            scale = 1.0
        return [cost / scale for cost in cross_costs]

    if normalize == "per_snapshot_median":
        out = []
        for cost, source_med, target_med in zip(cross_costs, source_medians, target_medians):
            scale = float(np.sqrt(max(source_med, 1e-12) * max(target_med, 1e-12)))
            if scale <= 0:
                scale = 1.0
            out.append(cost / scale)
        return out

    raise ValueError(f"Unknown normalization mode: {normalize}")


def uniform_weights(n):
    """
    Create uniform empirical weights.

    :param n: Number of points.
    :return: Uniform weight vector.
    """
    return np.ones(n, dtype=np.float64) / float(n)


def entropic_gw2(cost_1, cost_2, epsilon, max_iter, tol):
    """
    Compute entropic Gromov-Wasserstein squared distance.

    :param cost_1: Source internal cost matrix.
    :param cost_2: Target internal cost matrix.
    :param epsilon: Entropic regularization.
    :param max_iter: Maximum solver iterations.
    :param tol: Solver tolerance.
    :return: Squared GW value.
    """
    p = uniform_weights(cost_1.shape[0])
    q = uniform_weights(cost_2.shape[0])

    val = ot.gromov.entropic_gromov_wasserstein2(
        cost_1,
        cost_2,
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


def gw2(cost_1, cost_2, max_iter, tol):
    """
    Compute non-entropic Gromov-Wasserstein squared distance.

    This is slower than entropic GW and should first be used on smaller
    landmark counts as a higher-fidelity calibration reference.

    :param cost_1: Source internal cost matrix.
    :param cost_2: Target internal cost matrix.
    :param max_iter: Maximum solver iterations.
    :param tol: Solver tolerance.
    :return: Squared GW value.
    """
    p = uniform_weights(cost_1.shape[0])
    q = uniform_weights(cost_2.shape[0])

    val = ot.gromov.gromov_wasserstein2(
        cost_1,
        cost_2,
        p,
        q,
        loss_fun="square_loss",
        max_iter=max_iter,
        tol_rel=tol,
        tol_abs=tol,
        verbose=False,
    )

    return float(val)


def fused_gw2(cross_cost, cost_1, cost_2, alpha, epsilon, max_iter, tol):
    """
    Compute fused Gromov-Wasserstein squared distance.

    :param cross_cost: Feature-space cost matrix between source and target.
    :param cost_1: Source internal cost matrix.
    :param cost_2: Target internal cost matrix.
    :param alpha: FGW tradeoff parameter. Values near 1 emphasize relational GW.
    :param epsilon: Entropic regularization. If non-positive, use non-entropic FGW.
    :param max_iter: Maximum solver iterations.
    :param tol: Solver tolerance.
    :return: Squared FGW value.
    """
    p = uniform_weights(cost_1.shape[0])
    q = uniform_weights(cost_2.shape[0])

    if epsilon is not None and epsilon > 0:
        val = ot.gromov.entropic_fused_gromov_wasserstein2(
            cross_cost,
            cost_1,
            cost_2,
            p,
            q,
            loss_fun="square_loss",
            alpha=alpha,
            epsilon=epsilon,
            max_iter=max_iter,
            tol=tol,
            verbose=False,
        )
    else:
        val = ot.gromov.fused_gromov_wasserstein2(
            cross_cost,
            cost_1,
            cost_2,
            p,
            q,
            loss_fun="square_loss",
            alpha=alpha,
            max_iter=max_iter,
            tol_rel=tol,
            tol_abs=tol,
            verbose=False,
        )

    return float(val)


def wasserstein_identity2(cross_cost):
    """
    Compute a simple identity-matched mean squared displacement proxy.

    This is not an optimized transport solve. It assumes the landmark index set
    is shared across checkpoints and measures direct coordinate displacement.

    :param cross_cost: Cross-cost matrix between corresponding landmark clouds.
    :return: Mean diagonal cost squared proxy.
    """
    diag = np.diag(cross_cost)
    return float(np.mean(diag * diag))


def sliced_wasserstein2(x_1, x_2, n_projections, seed):
    """
    Compute a simple sliced-Wasserstein squared proxy.

    This implementation uses random one-dimensional projections and uniform
    empirical weights. It is included as a lightweight baseline and does not
    require POT solver calls.

    :param x_1: Source landmark cloud.
    :param x_2: Target landmark cloud.
    :param n_projections: Number of random projections.
    :param seed: Random seed.
    :return: Mean squared one-dimensional Wasserstein proxy.
    """
    rng = np.random.default_rng(seed)
    dim = x_1.shape[1]
    vals = []

    for _ in range(int(n_projections)):
        direction = rng.normal(size=dim)
        norm = np.linalg.norm(direction)
        if norm <= 0:
            continue
        direction = direction / norm

        p1 = np.sort(x_1 @ direction)
        p2 = np.sort(x_2 @ direction)
        vals.append(float(np.mean((p1 - p2) ** 2)))

    if not vals:
        return float("nan")

    return float(np.mean(vals))


def solve_ot_distance(method, x_1, x_2, cost_1, cost_2, cross_cost, epsilon, alpha, max_iter, tol, n_projections, seed):
    """
    Dispatch to an OT or OT-like distance solver.

    :param method: OT method name.
    :param x_1: Source landmark cloud.
    :param x_2: Target landmark cloud.
    :param cost_1: Source internal cost matrix.
    :param cost_2: Target internal cost matrix.
    :param cross_cost: Source-target feature cost matrix.
    :param epsilon: Entropic regularization.
    :param alpha: FGW tradeoff parameter.
    :param max_iter: Maximum solver iterations.
    :param tol: Solver tolerance.
    :param n_projections: Number of projections for sliced Wasserstein.
    :param seed: Random seed.
    :return: Solver result dictionary.
    """
    try:
        if method == "entropic_gw":
            value = entropic_gw2(cost_1, cost_2, epsilon, max_iter, tol)

        elif method == "gw":
            value = gw2(cost_1, cost_2, max_iter, tol)

        elif method == "fused_gw":
            value = fused_gw2(cross_cost, cost_1, cost_2, alpha, epsilon, max_iter, tol)

        elif method == "sliced_wasserstein":
            value = sliced_wasserstein2(x_1, x_2, n_projections, seed)

        elif method == "wasserstein_identity":
            value = wasserstein_identity2(cross_cost)

        else:
            raise ValueError(f"Unknown OT method: {method}")

        return {
            "ot_value": float(value),
            "ot_distance": float(np.sqrt(max(float(value), 0.0))),
            "ot_status": "ok",
            "ot_error": None,
        }

    except Exception as exc:
        return {
            "ot_value": float("nan"),
            "ot_distance": float("nan"),
            "ot_status": "failed",
            "ot_error": repr(exc),
        }


def build_pairs(num_checkpoints, mode):
    """
    Build checkpoint comparison pairs.

    :param num_checkpoints: Number of checkpoints.
    :param mode: Pairing mode.
    :return: List of index pairs.
    """
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


def write_dataframe(df, output):
    """
    Atomically write a dataframe to parquet or CSV.

    :param df: Dataframe to write.
    :param output: Output path ending in ``.parquet`` or ``.csv``.
    """
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


def parse_args():
    """
    Parse CLI arguments.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compute OT distances along one parquet checkpoint trajectory."
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
        help="adjacent: X_t to X_{t+1}; from_start: X_0 to X_t; full: all pairs.",
    )

    parser.add_argument(
        "--ot-method",
        choices=[
            "entropic_gw",
            "gw",
            "fused_gw",
            "sliced_wasserstein",
            "wasserstein_identity",
        ],
        default="entropic_gw",
        help="Optimal-transport method to compute.",
    )

    parser.add_argument(
        "--landmark-method",
        choices=["random", "fps", "stratified_label", "stratified_cluster"],
        default="random",
        help="Landmark selection strategy applied to the first checkpoint.",
    )

    parser.add_argument("--label-col", default="label".
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
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--n-projections", type=int, default=128)

    return parser.parse_args()


def validate_args(args):
    """
    Validate parsed arguments.

    :param args: Parsed arguments.
    """
    if args.n_landmarks <= 0:
        raise ValueError("--n-landmarks must be positive")

    if args.max_iter <= 0:
        raise ValueError("--max-iter must be positive")

    if args.tol <= 0:
        raise ValueError("--tol must be positive")

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")

    if args.n_projections <= 0:
        raise ValueError("--n-projections must be positive")


def main():
    """
    Run the OT trajectory worker.
    """
    args = parse_args()
    validate_args(args)

    run_dir = Path(args.run_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
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
    print(f"[INFO] ot_method={args.ot_method}")
    print(f"[INFO] n_landmarks={args.n_landmarks}")
    print(f"[INFO] landmark_method={args.landmark_method}")
    print(f"[INFO] normalize={args.normalize}")

    first_df, first_epoch = load_checkpoint_dataframe(ckpt_paths[0])
    first_x = dataframe_to_point_cloud(first_df, path=ckpt_paths[0])

    landmark_idx = choose_landmarks(
        first_x,
        args.n_landmarks,
        args.seed,
        args.landmark_method,
        df=first_df,
        label_col=args.label_col,
    )

    first_coverage = landmark_coverage_stats(first_x, landmark_idx)

    checkpoint_info = []
    costs_raw = []
    clouds_lm = []
    coverage_rows = []
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
        cost = make_cost_matrix(x_lm, metric=args.metric)
        coverage = landmark_coverage_stats(x, landmark_idx)

        clouds_lm.append(x_lm)
        costs_raw.append(cost)
        coverage_rows.append(coverage)
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
        epoch_delta = int(target_epoch - source_epoch)

        print(
            f"[OT {pair_index}/{len(pairs)}] "

        print(
            f"[GW {pair_index}/{len(pairs)}] "
            f"epoch {source_epoch} -> {target_epoch}",
            flush=True,
        )

        source_med = float(snapshot_medians[i])
        target_med = float(snapshot_medians[j])

        if source_med > 0:
            median_distance_ratio = target_med / source_med
        else:
            median_distance_ratio = np.nan

        cross_raw = cdist(clouds_lm[i], clouds_lm[j], metric=args.metric).astype(
            np.float64,
            copy=False,
        )
        cross_cost = normalize_cross_costs(
            [cross_raw],
            args.normalize,
            common_scale,
            [source_med],
            [target_med],
        )[0]

        t0 = time.perf_counter()

        result = solve_ot_distance(
            args.ot_method,
            clouds_lm[i],
            clouds_lm[j],
            costs[i],
            costs[j],
            cross_cost,
            args.epsilon,
            args.alpha,
            args.max_iter,
            args.tol,
            args.n_projections,
            args.seed + pair_index,
        )

        elapsed = time.perf_counter() - t0
        ot_distance = result["ot_distance"]

        if np.isfinite(ot_distance):
            distance_per_epoch = float(ot_distance / max(epoch_delta, 1))
        else:
            distance_per_epoch = float("nan")

        row = {
            **meta,
            "source_index": int(i),
            "target_index": int(j),
            "source_epoch": int(source_epoch),
            "target_epoch": int(target_epoch),
            "epoch_delta": int(epoch_delta),
            "comparison_type": args.mode,
            "ot_method": args.ot_method,
            "ot_value": result["ot_value"],
            "ot_distance": result["ot_distance"],
            "ot_distance_per_epoch": distance_per_epoch,
            "ot_status": result["ot_status"],
            "ot_error": result["ot_error"],
            "gw2": result["ot_value"],
            "gw_distance": result["ot_distance"],
            "gw_distance_per_epoch": distance_per_epoch,
            "gw_time_sec": float(elapsed),
            "source_checkpoint_path": checkpoint_info[i]["path"],
            "target_checkpoint_path": checkpoint_info[j]["path"],
            "source_snapshot_median_distance": source_med,
            "target_snapshot_median_distance": target_med,
            "median_distance_ratio": float(median_distance_ratio),
            "normalization": args.normalize,
            "normalization_common_scale": common_scale,
            "n_landmarks": int(len(landmark_idx)),
            "landmark_seed": int(args.seed),
            "landmark_method": args.landmark_method,
            "distance_metric": args.metric,
            "epsilon": float(args.epsilon),
            "alpha": float(args.alpha),
            "max_iter": int(args.max_iter),
            "tol": float(args.tol),
            "n_projections": int(args.n_projections),
            "source_n_points": int(checkpoint_info[i]["n_points"]),
            "target_n_points": int(checkpoint_info[j]["n_points"]),
            "source_ambient_dim": int(checkpoint_info[i]["ambient_dim"]),
            "target_ambient_dim": int(checkpoint_info[j]["ambient_dim"]),
            "initial_landmark_coverage_mean": first_coverage["landmark_coverage_mean"],
            "initial_landmark_coverage_q95": first_coverage["landmark_coverage_q95"],
            "initial_landmark_coverage_max": first_coverage["landmark_coverage_max"],
        }

        for key, value in coverage_rows[i].items():
            row[f"source_{key}"] = value

        for key, value in coverage_rows[j].items():
            row[f"target_{key}"] = value

        rows.append(row)
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
        "ot_method": args.ot_method,
        "num_checkpoints": len(ckpt_paths),
        "checkpoint_paths": [str(path) for path in ckpt_paths],
        "epochs": [info["epoch"] for info in checkpoint_info],
        "n_landmarks": int(len(landmark_idx)),
        "landmark_seed": int(args.seed),
        "landmark_method": args.landmark_method,
        "label_col": args.label_col,
        "landmark_indices": landmark_idx.tolist(),
        "initial_landmark_coverage": first_coverage,
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
        "alpha": args.alpha,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "n_projections": args.n_projections,
        "distance_metric": args.metric,
        "max_iter": args.max_iter,
        "tol": args.tol,
    }

    meta_out.write_text(json.dumps(meta_payload, indent=2, sort_keys=True))

    print(f"[DONE] wrote {output}")
    print(f"[DONE] wrote {meta_out}")


if __name__ == "__main__":
    main()
