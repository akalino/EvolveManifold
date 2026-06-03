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
                min_unique_rows=("unique_rows_approx", "min"),
            )
            .reset_index()
        )

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


if __name__ == "__main__":
    main()
