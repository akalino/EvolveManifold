#!/usr/bin/env python3
"""
Generate Figure 2: representative EvolveManifold synthetic collapse trajectories.

Rows:
  1. Linear projection collapse
  2. Radial collapse
  3. Cluster tightening
  4. Cluster merging
  5. Hole filling

Columns:
  Initial / Middle / Final checkpoints

Each row uses a separate PCA fit jointly over its three checkpoints.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


CKPT_RE = re.compile(r"ckpt_epoch_(\d+)\.pkl$")

COLORBLIND_PALETTE = np.array([
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
])


PANEL_CONFIGS = [
    {
        "row_title": "Projection collapse",
        "row_subtitle": "Spiked Gaussian \u2192 low-dimensional subspace",
        "mechanism": "linear_to_kplane",
        "model": "spiked_gaussian_n1000_d100_k33__linear__strong__mp1.0__noise0.0__seed5",
    },
    {
        "row_title": "Radial collapse",
        "row_subtitle": "Torus contracts toward its centroid",
        "mechanism": "radial_collapse",
        "model": "torus_n1000_d100_k33__linear__strong__mp1.0__noise0.0__seed5",
    },
    {
        "row_title": "Cluster tightening",
        "row_subtitle": "Within-cluster spread contracts",
        "mechanism": "cluster_tightening",
        "model": "clustered_gaussian_n1000_d100_k33__linear__strong__mp1.0__noise0.0__seed5",
    },
    {
        "row_title": "Cluster merging",
        "row_subtitle": "Cluster centers move together",
        "mechanism": "cluster_merging",
        "model": "clustered_gaussian_n1000_d100_k33__linear__strong__mp1.0__noise0.0__seed5",
    },
    {
        "row_title": "Hole filling",
        "row_subtitle": "Torus loses its central void",
        "mechanism": "hole_fill",
        "model": "torus_n1000_d100_k33__linear__strong__mp1.0__noise0.0__seed5",
    },
]


def load_checkpoint(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def checkpoint_paths_for_run(run_dir: Path) -> list[Path]:
    pairs = []
    for path in run_dir.iterdir():
        m = CKPT_RE.match(path.name)
        if m:
            pairs.append((int(m.group(1)), path))
    pairs.sort(key=lambda z: z[0])
    return [p for _, p in pairs]


def label_colors(labels: np.ndarray) -> list[str]:
    labels = np.asarray(labels)
    unique = np.unique(labels)
    label_to_color = {
        lab: COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]
        for i, lab in enumerate(unique)
    }
    return [label_to_color[lab] for lab in labels]

def choose_initial_middle_final(paths: list[Path]) -> list[Path]:
    if len(paths) < 3:
        raise ValueError(f"Need at least 3 checkpoints, found {len(paths)}")
    return [paths[0], paths[len(paths) // 2], paths[-1]]


def get_labels(payloads: list[dict]) -> np.ndarray | None:
    for payload in payloads:
        for key in ["cluster_labels", "labels", "y"]:
            if key in payload:
                labels = np.asarray(payload[key])
                if labels.ndim == 1 and labels.shape[0] == payload["x"].shape[0]:
                    return labels
    return None


def joint_pca(xs: list[np.ndarray]) -> tuple[list[np.ndarray], tuple[float, float]]:
    x_all = np.vstack(xs)
    pca = PCA(n_components=2, random_state=0)
    z_all = pca.fit_transform(x_all)

    splits = np.cumsum([x.shape[0] for x in xs])[:-1]
    zs = list(np.split(z_all, splits))

    evr = pca.explained_variance_ratio_
    return zs, (float(evr[0]), float(evr[1]))


def shared_limits(zs: list[np.ndarray], pad_frac: float = 0.07):
    z_all = np.vstack(zs)
    x_min, x_max = float(z_all[:, 0].min()), float(z_all[:, 0].max())
    y_min, y_max = float(z_all[:, 1].min()), float(z_all[:, 1].max())

    x_pad = pad_frac * max(x_max - x_min, 1e-8)
    y_pad = pad_frac * max(y_max - y_min, 1e-8)

    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def plot_panel(
    ax,
    z: np.ndarray,
    labels: np.ndarray | None,
    title: str,
    point_size: float,
    alpha: float,
):
    if labels is not None:
        ax.scatter(
            z[:, 0],
            z[:, 1],
            c=label_colors(labels),
            s=point_size,
            alpha=max(alpha, 0.75),
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

    ax.set_title(title, fontsize=9, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_alpha(0.35)


def make_figure(
    checkpoint_root: Path,
    out_path: Path,
    experiment: str,
    point_size: float,
    alpha: float,
    dpi: int,
):
    n_rows = len(PANEL_CONFIGS)
    n_cols = 3

    # Good for IEEE two-column when used as a full-width figure.
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7.1, 8.0),
        constrained_layout=False,
    )

    col_titles = ["Initial", "Middle", "Final"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=10, fontweight="bold", pad=6)

    for i, cfg in enumerate(PANEL_CONFIGS):
        run_dir = checkpoint_root / experiment / cfg["mechanism"] / cfg["model"]
        if not run_dir.exists():
            raise FileNotFoundError(f"Missing run directory: {run_dir}")

        paths = choose_initial_middle_final(checkpoint_paths_for_run(run_dir))
        payloads = [load_checkpoint(p) for p in paths]
        xs = [np.asarray(p["x"]) for p in payloads]
        epochs = [int(p.get("epoch", -1)) for p in payloads]
        labels = get_labels(payloads)

        zs, evr = joint_pca(xs)
        xlim, ylim = shared_limits(zs)

        for j, z in enumerate(zs):
            title = f"{col_titles[j]} (t={epochs[j]})"
            plot_panel(
                axes[i, j],
                z,
                labels=labels,
                title=title,
                point_size=point_size,
                alpha=alpha,
            )
            axes[i, j].set_xlim(*xlim)
            axes[i, j].set_ylim(*ylim)

        # Human-friendly row label at left.
        axes[i, 0].set_ylabel(
            f"{cfg['row_title']}\n{cfg['row_subtitle']}",
            fontsize=8.5,
            rotation=0,
            ha="right",
            va="center",
            labelpad=52,
        )

        # Small PCA note on the final panel.
        axes[i, 2].text(
            0.98,
            0.02,
            f"PC var. {evr[0]:.2f}, {evr[1]:.2f}",
            transform=axes[i, 2].transAxes,
            ha="right",
            va="bottom",
            fontsize=6.5,
            alpha=0.7,
        )

    fig.subplots_adjust(
        left=0.23,
        right=0.985,
        top=0.955,
        bottom=0.045,
        wspace=0.08,
        hspace=0.32,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"[WROTE] {out_path}")

    # Also write a PDF version for LaTeX.
    if out_path.suffix.lower() != ".pdf":
        pdf_path = out_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"[WROTE] {pdf_path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-root",
        default="/media/alex/WD_BLACK/evolve_collapse/evolve_checkpoints",
        help="Root directory containing checkpoint trajectories.",
    )
    parser.add_argument(
        "--experiment",
        default="collapse_ph",
        help="Experiment folder name under checkpoint-root.",
    )
    parser.add_argument(
        "--out",
        default="/media/alex/WD_BLACK/evolve_collapse/summary_assets/paper_figures/fig02_representative_trajectories.png",
        help="Output figure path.",
    )
    parser.add_argument("--point-size", type=float, default=2.2)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--dpi", type=int, default=300)

    args = parser.parse_args()

    make_figure(
        checkpoint_root=Path(args.checkpoint_root),
        out_path=Path(args.out),
        experiment=args.experiment,
        point_size=args.point_size,
        alpha=args.alpha,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
