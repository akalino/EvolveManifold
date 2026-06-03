#!/usr/bin/env python3
"""
Generate PCA trajectory panels for EvolveManifold checkpoints.

Each output image shows early, middle, and late checkpoints for one generated
trajectory. PCA is fit jointly on the selected checkpoints so that visual
movement across panels is meaningful.

Typical usage:

python generate_trajectory_panels.py \
  --checkpoint-root /media/alkal/WD_BLACK/evolve_collapse/evolve_checkpoints \
  --out-dir /media/alkal/WD_BLACK/evolve_collapse/summary_assets/candidate_trajectory_panels \
  --mechanisms cluster_merging cluster_tightening hole_fill linear_to_kplane radial_collapse \
  --dims 50 100 \
  --n-values 1000 \
  --max-per-mechanism 25
"""

import argparse
import os
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from projection_visualizers import AVAILABLE_PROJECTIONS, project_joint_snapshots


CKPT_RE = re.compile(r"ckpt_epoch_(\d+)\.pkl$")


def load_checkpoint(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def find_run_dirs(checkpoint_root: Path):
    run_dirs = []
    for dirpath, _, filenames in os.walk(checkpoint_root):
        if any(CKPT_RE.match(name) for name in filenames):
            run_dirs.append(Path(dirpath))
    return sorted(run_dirs)


def checkpoint_paths_for_run(run_dir):
    pairs = []
    for path in Path(run_dir).glob("ckpt_epoch_*.pkl"):
        m = re.search(r"ckpt_epoch_(\d+)\.pkl$", path.name)
        if m:
            pairs.append((int(m.group(1)), path))
    pairs.sort(key=lambda z: z[0])
    return [p for _, p in pairs]


def choose_early_mid_late(paths):
    if len(paths) < 3:
        return paths

    idxs = [0, len(paths) // 2, len(paths) - 1]
    return [paths[i] for i in idxs]


def parse_run_metadata(run_dir: Path):
    """
    Works with the current nested structure:
      .../<experiment>/<mechanism>/<model>/ckpt_epoch_XXXX.pkl

    The parser is deliberately permissive. If a field cannot be found, it
    remains None rather than breaking plotting.
    """
    parts = run_dir.parts

    experiment = parts[-3] if len(parts) >= 3 else "unknown_experiment"
    mechanism = parts[-2] if len(parts) >= 2 else "unknown_mechanism"
    model = parts[-1] if len(parts) >= 1 else "unknown_model"

    def find_int(pattern, text):
        m = re.search(pattern, text)
        return int(m.group(1)) if m else None

    def find_float(pattern, text):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    # Common model-name fields.
    n = find_int(r"n(\d+)", model)
    d = find_int(r"d(\d+)", model)
    k = find_int(r"k(\d+)", model)
    seed = find_int(r"seed(\d+)", model)
    mover_frac = find_float(r"mp([0-9.]+)", model)
    noise = find_float(r"noise([0-9.]+)", model)

    # If schedule/severity are encoded in experiment or model names, this may catch them.
    joined = "__".join([experiment, mechanism, model])

    schedule = None
    for candidate in ["linear", "sigmoid", "exponential"]:
        if candidate in joined:
            schedule = candidate
            break

    severity = None
    for candidate in ["weak", "moderate", "medium", "strong"]:
        if candidate in joined:
            severity = candidate
            break

    return {
        "experiment": experiment,
        "mechanism": mechanism,
        "model": model,
        "n": n,
        "d": d,
        "k": k,
        "seed": seed,
        "mover_frac": mover_frac,
        "noise": noise,
        "schedule": schedule,
        "severity": severity,
    }


def run_matches(meta, mechanisms, dims, n_values, seeds, severities, mover_fracs):
    if mechanisms and meta["mechanism"] not in mechanisms:
        return False

    if dims and meta["d"] not in dims:
        return False

    if n_values and meta["n"] not in n_values:
        return False

    if seeds and meta["seed"] not in seeds:
        return False

    if severities and meta.get("severity") not in severities:
        return False

    if mover_fracs:
        mf = meta.get("mover_frac")
        if mf is None:
            return False
        if not any(abs(float(mf) - float(target)) < 1e-9 for target in mover_fracs):
            return False

    return True


def find_labels_for_run(run_dir: Path, payloads):
    """
    Prefer labels embedded in checkpoint payloads. If not available, return None.

    This supports common keys:
      labels, y, cluster_labels
    """
    for payload in payloads:
        for key in ["labels", "y", "cluster_labels"]:
            if key in payload:
                labels = np.asarray(payload[key])
                if labels.ndim == 1 and len(labels) == len(payload["x"]):
                    return labels
    return None


def safe_filename(text):
    text = str(text)
    text = text.replace(os.sep, "_")
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def make_title(meta):
    pieces = [
        f"geom={meta['experiment']}",
        f"mech={meta['mechanism']}",
    ]

    for key in ["n", "d", "k", "seed", "schedule", "severity", "mover_frac", "noise"]:
        val = meta.get(key)
        if val is not None:
            pieces.append(f"{key}={val}")

    return " | ".join(pieces)


def plot_run_panel(args, run_dir: Path, out_dir: Path, point_size: float, alpha: float):
    meta = parse_run_metadata(run_dir)
    ckpt_paths = checkpoint_paths_for_run(run_dir)
    selected_paths = choose_early_mid_late(ckpt_paths)

    if len(selected_paths) < 2:
        print(f"[SKIP] not enough checkpoints: {run_dir}")
        return None

    payloads = [load_checkpoint(p) for p in selected_paths]
    xs = [np.asarray(payload["x"]) for payload in payloads]
    epochs = [payload.get("epoch", None) for payload in payloads]

    # Fit PCA jointly so early/mid/late are in the same coordinate system.
    x_all = np.vstack(xs)

    if x_all.shape[1] < 2:
        print(f"[SKIP] dimension < 2: {run_dir}")
        return None

    labels = find_labels_for_run(run_dir, payloads)

    result = project_joint_snapshots(
        xs,
        method=args.projection,
        labels=labels,
        seed=0,
    )

    zs = result.zs
    z_all = np.vstack(zs)
    projection_subtitle = result.subtitle
    projection_method = result.method

    splits = np.cumsum([len(x) for x in xs])[:-1]
    zs = np.split(z_all, splits)


    fig, axes = plt.subplots(1, len(zs), figsize=(4.2 * len(zs), 4.0), constrained_layout=True)

    if len(zs) == 1:
        axes = [axes]

    # Shared limits make trajectory changes easier to compare.
    x_min, x_max = z_all[:, 0].min(), z_all[:, 0].max()
    y_min, y_max = z_all[:, 1].min(), z_all[:, 1].max()

    x_pad = 0.05 * max(x_max - x_min, 1e-8)
    y_pad = 0.05 * max(y_max - y_min, 1e-8)

    for ax, z, epoch in zip(axes, zs, epochs):
        if labels is not None and len(labels) == len(z):
            ax.scatter(
                z[:, 0],
                z[:, 1],
                c=labels,
                s=point_size,
                alpha=alpha,
                linewidths=0,
            )
        else:
            ax.scatter(
                z[:, 0],
                z[:, 1],
                s=point_size,
                alpha=alpha,
                linewidths=0,
            )

        label = f"epoch {epoch}" if epoch is not None else "checkpoint"
        ax.set_title(label)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_aspect("equal", adjustable="box")

    # explained = pca.explained_variance_ratio_
    fig.suptitle(
        make_title(meta) + f"\nProjection: {projection_subtitle}",
        fontsize=10,
    )

    mechanism_dir = out_dir / safe_filename(meta["mechanism"])
    mechanism_dir.mkdir(parents=True, exist_ok=True)

    filename = safe_filename(
        f"{projection_method}__{meta['experiment']}__{meta['mechanism']}__{meta['model']}"
    ) + ".png"

    out_path = mechanism_dir / filename
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate PCA early/middle/late trajectory panels."
    )
    parser.add_argument(
        "--checkpoint-root",
        required=True,
        help="Root directory containing checkpointed trajectory runs.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for candidate trajectory panels.",
    )
    parser.add_argument(
        "--mechanisms",
        nargs="*",
        default=None,
        help="Optional mechanism filter.",
    )
    parser.add_argument(
        "--dims",
        nargs="*",
        type=int,
        default=None,
        help="Optional ambient dimension filter.",
    )
    parser.add_argument(
        "--n-values",
        nargs="*",
        type=int,
        default=None,
        help="Optional sample-size filter.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Optional seed filter.",
    )
    parser.add_argument(
        "--max-per-mechanism",
        type=int,
        default=25,
        help="Maximum number of panels to generate per mechanism.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=4.0,
        help="Scatter point size.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.75,
        help="Scatter alpha.",
    )
    parser.add_argument(
        "--projection",
        default="raw01",
        choices=AVAILABLE_PROJECTIONS,
        help="Projection method for 2D trajectory panels.",
    )
    parser.add_argument(
        "--severities",
        nargs="*",
        default=None,
        help="Optional severity filter, e.g. strong moderate weak.",
    )
    parser.add_argument(
        "--mover-fracs",
        nargs="*",
        type=float,
        default=None,
        help="Optional mover-fraction filter, e.g. 1.0 0.25.",
    )

    args = parser.parse_args()

    checkpoint_root = Path(args.checkpoint_root)
    out_dir = Path(args.out_dir)

    run_dirs = find_run_dirs(checkpoint_root)

    counts = {}
    written = []

    for run_dir in run_dirs:
        meta = parse_run_metadata(run_dir)

        if not run_matches(
            meta,
            mechanisms=args.mechanisms,
            dims=args.dims,
            n_values=args.n_values,
            seeds=args.seeds,
            severities=args.severities,
            mover_fracs=args.mover_fracs
        ):
            continue

        mechanism = meta["mechanism"]
        counts.setdefault(mechanism, 0)

        if counts[mechanism] >= args.max_per_mechanism:
            continue

        try:
            out_path = plot_run_panel(
                args=args,
                run_dir=run_dir,
                out_dir=out_dir,
                point_size=args.point_size,
                alpha=args.alpha,
            )
            if out_path is not None:
                written.append(out_path)
                counts[mechanism] += 1
                print(f"[WROTE] {out_path}")
        except Exception as exc:
            print(f"[WARN] failed for {run_dir}: {exc}")

    print("\n[DONE]")
    print(f"Candidate panels written: {len(written)}")
    for mechanism, count in sorted(counts.items()):
        print(f"  {mechanism}: {count}")


if __name__ == "__main__":
    main()