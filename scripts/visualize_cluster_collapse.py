import argparse
import glob
import os
import re
import pickle
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


import umap
HAS_UMAP = True



def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize clustered collapse from checkpoints and saved labels."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing checkpoint .npy files.",
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        required=True,
        help="Directory containing saved *_labels.npy files.",
    )
    parser.add_argument(
        "--run_stem",
        type=str,
        required=True,
        help="Run stem, e.g. clustered_gaussian_n5000_d50_k16__linear__moderate__mp1.0__noise0.0__seed17",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Where to save figures.",
    )
    parser.add_argument(
        "--epochs",
        type=str,
        default="0,10,25,49",
        help="Comma-separated epochs to visualize, e.g. 0,10,25,49",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="pca",
        choices=["pca", "tsne", "umap"],
        help="Embedding method.",
    )
    parser.add_argument(
        "--align_pca_globally",
        action="store_true",
        help="For PCA, fit one PCA model on the stacked selected checkpoints for temporal compatibility.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="TSNE perplexity.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=0,
        help="Random seed for PCA/TSNE/UMAP.",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=8.0,
        help="Scatter point size.",
    )
    return parser.parse_args()


def parse_epoch_list(epoch_str: str) -> List[int]:
    return [int(x.strip()) for x in epoch_str.split(",") if x.strip()]


def checkpoint_sort_key(path: str) -> int:
    fname = os.path.basename(path)
    m = re.search(r"^ckpt_epoch_(\d+)\.pkl$", fname)
    if m is None:
        raise ValueError(f"Could not parse epoch from filename: {fname}")
    return int(m.group(1))


def load_checkpoints_for_run(checkpoint_dir: str, run_stem: str):
    run_stem = run_stem.rstrip("/")

    run_dir = os.path.join(checkpoint_dir, run_stem)
    pattern = os.path.join(run_dir, "ckpt_epoch_*.pkl")
    paths = sorted(glob.glob(pattern), key=checkpoint_sort_key)

    if not paths:
        raise ValueError(f"No checkpoint files found in {run_dir}")

    out = []
    for path in paths:
        with open(path, "rb") as f:
            obj = pickle.load(f)

        if not isinstance(obj, dict):
            raise ValueError(f"Expected checkpoint dict, got {type(obj)} in {path}")

        if "x" not in obj:
            raise ValueError(f"Checkpoint missing 'x' key: {path}")

        x = np.asarray(obj["x"])
        print(
            path,
            "shape=", x.shape,
            "dtype=", x.dtype,
            "finite_frac=", np.isfinite(x).mean(),
            "nan_count=", np.isnan(x).sum(),
            "posinf_count=", np.isposinf(x).sum(),
            "neginf_count=", np.isneginf(x).sum(),
        )

        finite_mask = np.isfinite(x)
        if finite_mask.any():
            xf = x[finite_mask]
            print(
                "min=", xf.min(),
                "max=", xf.max(),
                "absmax=", np.abs(xf).max(),
                "mean=", xf.mean(),
                "std=", xf.std(),
            )

        if x.ndim != 2:
            raise ValueError(f"Checkpoint x should be 2D, got shape {x.shape} in {path}")

        out.append((checkpoint_sort_key(path), x, path))

    return out


def filter_epochs(checkpoints, requested_epochs):
    by_epoch = {epoch: (x, path) for epoch, x, path in checkpoints}
    selected = []
    for ep in requested_epochs:
        if ep not in by_epoch:
            raise ValueError(f"Requested epoch {ep} not found in checkpoints")
        x, path = by_epoch[ep]
        selected.append((ep, x, path))
    return selected


def load_labels(label_dir: str, run_stem: str) -> np.ndarray:
    run_stem = run_stem.rstrip("/")
    path = os.path.join(label_dir, f"{run_stem}_labels.npy")
    if not os.path.exists(path):
        raise ValueError(f"Label file not found: {path}")

    obj = np.load(path, allow_pickle=True)

    # Case 1: normal numeric array
    if isinstance(obj, np.ndarray) and obj.dtype != object:
        labels = np.asarray(obj).reshape(-1).astype(int)
        return labels

    # Case 2: 0-d object array, unwrap it
    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()

    # Unwrapped possibilities
    if isinstance(obj, dict):
        if "labels" in obj:
            labels = np.asarray(obj["labels"]).reshape(-1).astype(int)
            return labels
        raise ValueError(f"Label file contains dict without 'labels' key: {path}")

    if isinstance(obj, tuple) or isinstance(obj, list):
        # common accidental save shape: (x, labels)
        if len(obj) == 2:
            labels = np.asarray(obj[1]).reshape(-1).astype(int)
            return labels
        labels = np.asarray(obj).reshape(-1).astype(int)
        return labels

    labels = np.asarray(obj).reshape(-1).astype(int)
    return labels


def embed_pca_global(selected_checkpoints, random_state=0):
    xs = [x for _, x, _ in selected_checkpoints]
    stacked = np.vstack(xs)

    pca = PCA(n_components=2, random_state=random_state)
    pca.fit(stacked)

    embedded = []
    for epoch, x, path in selected_checkpoints:
        z = pca.transform(x)
        embedded.append((epoch, z, path))

    return embedded


def embed_pca_independent(selected_checkpoints, random_state=0):
    embedded = []
    for epoch, x, path in selected_checkpoints:
        pca = PCA(n_components=2, random_state=random_state)
        z = pca.fit_transform(x)
        embedded.append((epoch, z, path))
    return embedded


def embed_tsne_independent(selected_checkpoints, perplexity=30.0, random_state=0):
    embedded = []
    for epoch, x, path in selected_checkpoints:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            init="pca",
        )
        z = tsne.fit_transform(x)
        embedded.append((epoch, z, path))
    return embedded


def sanitize_points(x, clip_value=1e6):
    x = np.asarray(x, dtype=np.float64).copy()
    x[~np.isfinite(x)] = np.nan

    col_means = np.nanmean(x, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)

    nan_rows, nan_cols = np.where(np.isnan(x))
    if len(nan_rows) > 0:
        x[nan_rows, nan_cols] = col_means[nan_cols]

    x = np.clip(x, -clip_value, clip_value)
    return x


def standardize_points(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.maximum(sd, eps)
    return (x - mu) / sd


def pre_reduce_for_umap(x, pre_pca_dim=15, random_state=0):
    x = np.asarray(x, dtype=np.float64)
    pre_pca_dim = min(pre_pca_dim, x.shape[0], x.shape[1])
    if pre_pca_dim < 2:
        return x
    pca = PCA(n_components=pre_pca_dim, random_state=random_state)
    return pca.fit_transform(x)


def embed_umap_independent(selected_checkpoints, random_state=0, pre_pca_dim=15, n_neighbors=100, min_dist=0.1):
    if not HAS_UMAP:
        raise ImportError("umap-learn is not installed. Install it or use --method pca/tsne.")

    embedded = []
    for epoch, x, path in selected_checkpoints:
        x = sanitize_points(x)
        x = standardize_points(x)
        x = pre_reduce_for_umap(x, pre_pca_dim=pre_pca_dim, random_state=random_state)

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )
        z = reducer.fit_transform(x)
        embedded.append((epoch, z, path))

    return embedded


def save_single_plot(z, labels, title, out_path, point_size=8.0):
    plt.figure(figsize=(6, 6))
    plt.scatter(z[:, 0], z[:, 1], c=labels, s=point_size)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_grid(embedded, labels, out_path, method_name, point_size=8.0):
    n = len(embedded)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (epoch, z, _) in zip(axes, embedded):
        ax.scatter(z[:, 0], z[:, 1], c=labels, s=point_size)
        ax.set_title(f"{method_name.upper()} | epoch {epoch}")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    requested_epochs = parse_epoch_list(args.epochs)
    print(requested_epochs)
    checkpoints = load_checkpoints_for_run(args.checkpoint_dir, args.run_stem)
    selected = filter_epochs(checkpoints, requested_epochs)
    labels = load_labels(args.label_dir, args.run_stem)
    print("labels type:", type(labels), "shape:", getattr(labels, "shape", None), "dtype:",
          getattr(labels, "dtype", None))

    for epoch, x, _ in selected:
        if len(labels) != len(x):
            raise ValueError(
                f"Label length {len(labels)} does not match checkpoint size {len(x)} at epoch {epoch}"
            )

    if args.method == "pca":
        if args.align_pca_globally:
            embedded = embed_pca_global(selected, random_state=args.random_state)
            suffix = "pca_global"
        else:
            embedded = embed_pca_independent(selected, random_state=args.random_state)
            suffix = "pca_independent"

    elif args.method == "tsne":
        embedded = embed_tsne_independent(
            selected,
            perplexity=args.perplexity,
            random_state=args.random_state,
        )
        suffix = "tsne"

    elif args.method == "umap":
        embedded = embed_umap_independent(selected, random_state=args.random_state)
        suffix = "umap"

    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Save individual figures
    for epoch, z, _ in embedded:
        out_path = os.path.join(
            args.out_dir,
            f"{args.run_stem}__epoch{epoch:03d}__{suffix}.png",
        )
        title = f"{args.run_stem} | epoch {epoch} | {suffix}"
        save_single_plot(
            z,
            labels,
            title=title,
            out_path=out_path,
            point_size=args.point_size,
        )

    # Save a grid figure
    grid_path = os.path.join(
        args.out_dir,
        f"{args.run_stem}__{suffix}__grid.png",
    )
    save_grid(
        embedded,
        labels,
        out_path=grid_path,
        method_name=suffix,
        point_size=args.point_size,
    )

    print(f"[DONE] Wrote visualizations to {args.out_dir}")


if __name__ == "__main__":
    main()