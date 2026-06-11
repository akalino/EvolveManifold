#!/usr/bin/env python3
"""
Audit whether EvolveManifold checkpoint trajectories actually change over time.

This checks:
- checkpoint ordering
- x shapes
- displacement from first checkpoint
- consecutive displacement
- effective rank
- pairwise distance summaries
- coordinate variance
- whether t=0 is degenerate
"""

import argparse
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from tqdm import tqdm


CKPT_RE = re.compile(r"ckpt_epoch_(\d+)\.pkl$")


def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def find_run_dirs(root):
    root = Path(root)
    run_dirs = []
    for dirpath, _, filenames in os.walk(root):
        if any(CKPT_RE.match(name) for name in filenames):
            run_dirs.append(Path(dirpath))
    return sorted(run_dirs)


def checkpoint_paths_for_run(run_dir):
    pairs = []
    for name in os.listdir(run_dir):
        m = CKPT_RE.match(name)
        if m:
            pairs.append((int(m.group(1)), Path(run_dir) / name))
    pairs.sort(key=lambda z: z[0])
    return pairs


def parse_meta(run_dir):
    parts = Path(run_dir).parts

    experiment = parts[-3] if len(parts) >= 3 else "unknown_experiment"
    mechanism = parts[-2] if len(parts) >= 2 else "unknown_mechanism"
    model = parts[-1] if len(parts) >= 1 else "unknown_model"

    def find_int(pattern, text):
        m = re.search(pattern, text)
        return int(m.group(1)) if m else None

    def find_float(pattern, text):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    joined = "__".join([experiment, mechanism, model])

    severity = None
    for cand in ["weak", "moderate", "medium", "strong"]:
        if cand in joined:
            severity = cand
            break

    schedule = None
    for cand in ["linear", "sigmoid", "exponential"]:
        if cand in joined:
            schedule = cand
            break

    return {
        "run_dir": str(run_dir),
        "experiment": experiment,
        "mechanism": mechanism,
        "model": model,
        "n": find_int(r"n(\d+)", model),
        "d": find_int(r"d(\d+)", model),
        "k": find_int(r"k(\d+)", model),
        "seed": find_int(r"seed(\d+)", model),
        "mover_frac": find_float(r"mp([0-9.]+)", model),
        "noise": find_float(r"noise([0-9.]+)", model),
        "severity": severity,
        "schedule": schedule,
    }


def effective_rank(x, eps=1e-12):
    x = np.asarray(x)
    x = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    vals = s ** 2
    total = vals.sum()
    if total <= eps:
        return 0.0
    p = vals / total
    p = p[p > eps]
    return float(np.exp(-(p * np.log(p)).sum()))


def distance_summary(x, max_points=1000):
    x = np.asarray(x)
    if x.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(x.shape[0], size=max_points, replace=False)
        x = x[idx]

    if x.shape[0] < 2:
        return {
            "mean_dist": np.nan,
            "median_dist": np.nan,
            "q90_dist": np.nan,
        }

    d = pdist(x)
    return {
        "mean_dist": float(np.mean(d)),
        "median_dist": float(np.median(d)),
        "q90_dist": float(np.quantile(d, 0.90)),
    }


def extract_labels(payload):
    """
    Extract one label per point from a checkpoint payload when available.

    Supported keys:
    - labels
    - y
    - cluster_labels

    Returns None when labels are missing or malformed.
    """
    for key in ["labels", "y", "cluster_labels"]:
        if key in payload:
            labels = np.asarray(payload[key])
            if labels.ndim == 1:
                return labels.astype(int)
    return None


def cluster_distance_summary(x, labels, max_points_per_cluster=500):
    """
    Compute cluster-specific diagnostics.

    These are especially useful for distinguishing:
    - cluster_tightening: within-cluster distances should decrease
    - cluster_merging: centroid spread / between-centroid distance should decrease

    Returns NaNs when labels are unavailable or invalid.
    """
    x = np.asarray(x)

    nan_result = {
        "num_clusters": np.nan,
        "within_cluster_mean_dist": np.nan,
        "within_cluster_median_dist": np.nan,
        "within_cluster_q90_dist": np.nan,
        "cluster_centroid_spread": np.nan,
        "between_centroid_mean_dist": np.nan,
        "between_centroid_median_dist": np.nan,
        "between_centroid_q90_dist": np.nan,
    }

    if labels is None:
        return nan_result

    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.shape[0] != x.shape[0]:
        return nan_result

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        out = nan_result.copy()
        out["num_clusters"] = int(len(unique_labels))
        return out

    rng = np.random.default_rng(0)

    within_dists = []
    centroids = []

    for lab in unique_labels:
        idx = np.where(labels == lab)[0]
        if len(idx) == 0:
            continue

        x_lab = x[idx]
        centroids.append(x_lab.mean(axis=0))

        if len(x_lab) > max_points_per_cluster:
            sub_idx = rng.choice(len(x_lab), size=max_points_per_cluster, replace=False)
            x_lab_eval = x_lab[sub_idx]
        else:
            x_lab_eval = x_lab

        if len(x_lab_eval) >= 2:
            within_dists.append(pdist(x_lab_eval))

    if centroids:
        centroids = np.vstack(centroids)
    else:
        return nan_result

    if within_dists:
        within_all = np.concatenate(within_dists)
        within_mean = float(np.mean(within_all))
        within_median = float(np.median(within_all))
        within_q90 = float(np.quantile(within_all, 0.90))
    else:
        within_mean = np.nan
        within_median = np.nan
        within_q90 = np.nan

    centroid_center = centroids.mean(axis=0, keepdims=True)
    centroid_radii = np.linalg.norm(centroids - centroid_center, axis=1)
    centroid_spread = float(np.mean(centroid_radii))

    if len(centroids) >= 2:
        between = pdist(centroids)
        between_mean = float(np.mean(between))
        between_median = float(np.median(between))
        between_q90 = float(np.quantile(between, 0.90))
    else:
        between_mean = np.nan
        between_median = np.nan
        between_q90 = np.nan

    return {
        "num_clusters": int(len(unique_labels)),
        "within_cluster_mean_dist": within_mean,
        "within_cluster_median_dist": within_median,
        "within_cluster_q90_dist": within_q90,
        "cluster_centroid_spread": centroid_spread,
        "between_centroid_mean_dist": between_mean,
        "between_centroid_median_dist": between_median,
        "between_centroid_q90_dist": between_q90,
    }


def audit_run(run_dir):
    meta = parse_meta(run_dir)
    pairs = checkpoint_paths_for_run(run_dir)

    if len(pairs) == 0:
        return []

    payloads = []
    for epoch_from_name, path in pairs:
        payload = load_checkpoint(path)
        x = np.asarray(payload["x"])
        epoch_payload = payload.get("epoch", epoch_from_name)
        payloads.append((epoch_from_name, epoch_payload, path, x, payload))

    x0 = payloads[0][3]
    rows = []
    prev_x = None

    for i, (epoch_name, epoch_payload, path, x, payload) in enumerate(payloads):
        if x.shape != x0.shape:
            diff_from_first = np.nan
            rel_diff_from_first = np.nan
        else:
            diff_from_first = float(np.linalg.norm(x - x0))
            rel_diff_from_first = diff_from_first / (float(np.linalg.norm(x0)) + 1e-12)

        if prev_x is not None and prev_x.shape == x.shape:
            step_diff = float(np.linalg.norm(x - prev_x))
            rel_step_diff = step_diff / (float(np.linalg.norm(prev_x)) + 1e-12)
        else:
            step_diff = np.nan
            rel_step_diff = np.nan

        coord_std_mean = float(np.mean(np.std(x, axis=0)))
        coord_std_max = float(np.max(np.std(x, axis=0)))
        global_std = float(np.std(x))
        unique_rows_approx = int(np.unique(np.round(x, decimals=10), axis=0).shape[0])

        dist = distance_summary(x)
        labels = extract_labels(payload)
        cluster_dist = cluster_distance_summary(x, labels)

        row = {
            **meta,
            "checkpoint_index": i,
            "epoch_from_name": epoch_name,
            "epoch_payload": epoch_payload,
            "path": str(path),
            "x_shape": str(tuple(x.shape)),
            "x_min": float(np.min(x)),
            "x_max": float(np.max(x)),
            "x_mean": float(np.mean(x)),
            "x_std": global_std,
            "coord_std_mean": coord_std_mean,
            "coord_std_max": coord_std_max,
            "unique_rows_approx": unique_rows_approx,
            "fro_diff_from_first": diff_from_first,
            "rel_diff_from_first": rel_diff_from_first,
            "fro_step_diff": step_diff,
            "rel_step_diff": rel_step_diff,
            "effective_rank": effective_rank(x),
            **dist,
            **cluster_dist,
            "has_cluster_labels": labels is not None,
            "payload_keys": ",".join(sorted(payload.keys())),
        }
        rows.append(row)

        prev_x = x

    return rows


def matches(meta, mechanisms, dims, n_values, seeds, severities, mover_fracs):
    if mechanisms and meta["mechanism"] not in mechanisms:
        return False
    if dims and meta["d"] not in dims:
        return False
    if n_values and meta["n"] not in n_values:
        return False
    if seeds and meta["seed"] not in seeds:
        return False
    if severities and meta["severity"] not in severities:
        return False
    if mover_fracs:
        mf = meta["mover_frac"]
        if mf is None:
            return False
        if not any(abs(float(mf) - float(v)) < 1e-9 for v in mover_fracs):
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mechanisms", nargs="*", default=None)
    parser.add_argument("--dims", nargs="*", type=int, default=None)
    parser.add_argument("--n-values", nargs="*", type=int, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--severities", nargs="*", default=None)
    parser.add_argument("--mover-fracs", nargs="*", type=float, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()

    run_dirs = find_run_dirs(args.checkpoint_root)

    all_rows = []
    used = 0

    for run_dir in tqdm(run_dirs):
        meta = parse_meta(run_dir)

        if not matches(
            meta,
            mechanisms=args.mechanisms,
            dims=args.dims,
            n_values=args.n_values,
            seeds=args.seeds,
            severities=args.severities,
            mover_fracs=args.mover_fracs,
        ):
            continue

        rows = audit_run(run_dir)
        if rows:
            all_rows.extend(rows)
            used += 1

        if args.max_runs is not None and used >= args.max_runs:
            break

    df = pd.DataFrame(all_rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"[DONE] wrote {out}")
    print(f"runs audited: {used}")
    print(f"rows: {len(df)}")

    if len(df) > 0:
        summary = (
            df.groupby(["mechanism", "run_dir"], dropna=False)
            .agg(
                checkpoints=("checkpoint_index", "size"),
                first_epoch=("epoch_payload", "first"),
                last_epoch=("epoch_payload", "last"),
                max_rel_diff_from_first=("rel_diff_from_first", "max"),
                max_rel_step_diff=("rel_step_diff", "max"),
                first_x_std=("x_std", "first"),
                last_x_std=("x_std", "last"),
                first_eff_rank=("effective_rank", "first"),
                last_eff_rank=("effective_rank", "last"),
                first_mean_dist=("mean_dist", "first"),
                last_mean_dist=("mean_dist", "last"),
                first_within_cluster_mean_dist=("within_cluster_mean_dist", "first"),
                last_within_cluster_mean_dist=("within_cluster_mean_dist", "last"),
                first_cluster_centroid_spread=("cluster_centroid_spread", "first"),
                last_cluster_centroid_spread=("cluster_centroid_spread", "last"),
                first_between_centroid_mean_dist=("between_centroid_mean_dist", "first"),
                last_between_centroid_mean_dist=("between_centroid_mean_dist", "last"),
                min_unique_rows=("unique_rows_approx", "min"),
                has_cluster_labels=("has_cluster_labels", "max"),
            )
            .reset_index()
        )

        for num_col, den_col, out_col in [
            (
                "last_within_cluster_mean_dist",
                "first_within_cluster_mean_dist",
                "within_cluster_mean_dist_ratio",
            ),
            (
                "last_cluster_centroid_spread",
                "first_cluster_centroid_spread",
                "cluster_centroid_spread_ratio",
            ),
            (
                "last_between_centroid_mean_dist",
                "first_between_centroid_mean_dist",
                "between_centroid_mean_dist_ratio",
            ),
        ]:
            summary[out_col] = summary[num_col] / (summary[den_col].abs() + 1e-12)

        summary_path = out.with_name(out.stem + "_summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"[DONE] wrote {summary_path}")

        print("\nPotentially static runs:")
        static = summary[summary["max_rel_diff_from_first"] < 1e-8]
        print(static[["mechanism", "run_dir", "max_rel_diff_from_first"]].head(20))

        print("\nPotentially degenerate initial clouds:")
        deg = summary[
            (summary["first_x_std"] < 1e-8)
            | (summary["min_unique_rows"] <= 2)
        ]
        print(deg[["mechanism", "run_dir", "first_x_std", "min_unique_rows"]].head(20))

        print("\nCluster-mechanism diagnostics:")
        cluster_diag = summary[
            summary["mechanism"].isin(["cluster_tightening", "cluster_merging"])
        ]
        cluster_cols = [
            "mechanism",
            "run_dir",
            "has_cluster_labels",
            "first_within_cluster_mean_dist",
            "last_within_cluster_mean_dist",
            "within_cluster_mean_dist_ratio",
            "first_cluster_centroid_spread",
            "last_cluster_centroid_spread",
            "cluster_centroid_spread_ratio",
        ]
        if len(cluster_diag) > 0:
            print(cluster_diag[cluster_cols].head(30))
        else:
            print("No cluster_tightening or cluster_merging runs found.")


if __name__ == "__main__":
    main()
