"""
Compute GW distance from each checkpoint to simple reference templates.

Starter templates include an isotropic Gaussian reference, a collapsed cloud,
and a low-rank projection of the initial checkpoint.

This script is intentionally simple. It is meant to establish the template
distance workflow before adding more mathematically meaningful references.

Example
-------
python compute_gw_to_templates.py \\
    --checkpoint-files run/checkpoints/ckpt_epoch_*.parquet \\
    --output "$EVOLVE_ROOT/gw_outputs/example__template_distances.parquet" \\
    --n-landmarks 256
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import ot


CKPT_RE = re.compile(r"ckpt_epoch_(\d+)\.parquet$")


def epoch_from_path(path):
    """
    Parse checkpoint epoch from a checkpoint path.

    :param path: Checkpoint path.
    :return: Integer epoch, or -1 if unavailable.
    """
    match = CKPT_RE.match(path.name)
    if match is None:
        return -1
    return int(match.group(1))


def load_x(path, dim_cols=None):
    """
    Load coordinate matrix from a parquet checkpoint.

    :param path: Path to parquet checkpoint.
    :param dim_cols: Optional coordinate columns to reuse.
    :return: Pair ``(x, dim_cols)``.
    """
    df = pd.read_parquet(path)

    if dim_cols is None:
        dim_cols = sorted([c for c in df.columns if str(c).startswith("dim_")])

    if dim_cols:
        x = df[dim_cols].to_numpy(dtype=float, copy=True)
    else:
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            raise ValueError(f"No numeric coordinate columns found in {path}")
        dim_cols = list(numeric.columns)
        x = numeric.to_numpy(dtype=float, copy=True)

    return x, dim_cols


def make_cost(x):
    """
    Build an internal distance matrix for GW.

    :param x: Point cloud.
    :return: Pairwise Euclidean distance matrix.
    """
    return cdist(x, x, metric="euclidean").astype(np.float64, copy=False)


def uniform(n):
    """
    Create uniform empirical weights.

    :param n: Number of points.
    :return: Uniform weight vector.
    """
    return np.ones(n, dtype=float) / float(n)


def positive_median(cost):
    """
    Compute the median of positive entries in a cost matrix.

    :param cost: Cost matrix.
    :return: Positive median, with a safe fallback.
    """
    positive = cost[cost > 0]
    if positive.size == 0:
        return 1.0

    scale = float(np.median(positive))
    if not np.isfinite(scale) or scale <= 0:
        return 1.0

    return scale


def entropic_gw_distance(c1, c2, epsilon, max_iter, tol):
    """
    Compute entropic GW distance between two cost matrices.

    :param c1: Source internal cost matrix.
    :param c2: Target internal cost matrix.
    :param epsilon: Entropic regularization.
    :param max_iter: Maximum solver iterations.
    :param tol: Solver tolerance.
    :return: Square-rooted GW value.
    """
    val = ot.gromov.entropic_gromov_wasserstein2(
        c1,
        c2,
        uniform(c1.shape[0]),
        uniform(c2.shape[0]),
        loss_fun="square_loss",
        epsilon=epsilon,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
    )

    return float(np.sqrt(max(float(val), 0.0)))


def template_isotropic_gaussian(x0, seed):
    """
    Build an isotropic Gaussian template with the same shape as ``x0``.

    :param x0: Initial landmark cloud.
    :param seed: Random seed.
    :return: Template point cloud.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(size=x0.shape)


def template_collapsed_cloud(x0):
    """
    Build a fully collapsed point-cloud template.

    :param x0: Initial landmark cloud.
    :return: Collapsed point cloud at the initial centroid.
    """
    centroid = x0.mean(axis=0, keepdims=True)
    return np.repeat(centroid, repeats=x0.shape[0], axis=0)


def template_low_rank_projection(x0, rank):
    """
    Build a low-rank PCA projection template from the initial cloud.

    :param x0: Initial landmark cloud.
    :param rank: Projection rank.
    :return: Low-rank projected template.
    """
    center = x0.mean(axis=0, keepdims=True)
    xc = x0 - center

    u, svals, vt = np.linalg.svd(xc, full_matrices=False)

    rank = min(int(rank), vt.shape[0])
    projected = (u[:, :rank] * svals[:rank]) @ vt[:rank, :]

    return projected + center


def choose_landmarks(x, n_landmarks, seed):
    """
    Choose random landmark indices from the initial checkpoint.

    :param x: Initial point cloud.
    :param n_landmarks: Requested number of landmarks.
    :param seed: Random seed.
    :return: Sorted landmark indices.
    """
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    m = min(int(n_landmarks), int(n))

    return np.sort(rng.choice(n, size=m, replace=False))


def build_templates(x0, seed, low_rank):
    """
    Build all starter templates.

    :param x0: Initial landmark cloud.
    :param seed: Random seed.
    :param low_rank: Rank for low-rank projection template.
    :return: Dictionary mapping template names to point clouds.
    """
    return {
        "isotropic_gaussian": template_isotropic_gaussian(x0, seed),
        "collapsed_cloud": template_collapsed_cloud(x0),
        "low_rank_projection": template_low_rank_projection(x0, low_rank),
    }


def main():
    """
    Run the GW-to-template CLI.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-files", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-landmarks", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--low-rank", type=int, default=5)
    parser.add_argument(
        "--normalize",
        choices=["start_median", "none"],
        default="start_median",
    )
    args = parser.parse_args()

    checkpoint_files = sorted(
        [Path(p).expanduser().resolve() for p in args.checkpoint_files],
        key=epoch_from_path,
    )

    x0_full, dim_cols = load_x(checkpoint_files[0])
    landmark_idx = choose_landmarks(x0_full, args.n_landmarks, args.seed)

    x0 = x0_full[landmark_idx]
    templates = build_templates(x0, args.seed, args.low_rank)

    template_costs = {}
    for name, template_x in templates.items():
        template_costs[name] = make_cost(template_x)

    if args.normalize == "start_median":
        scale = positive_median(make_cost(x0))
    else:
        scale = 1.0

    rows = []

    for ckpt in checkpoint_files:
        print(f"[CHECKPOINT] {ckpt.name}")

        x_full, _ = load_x(ckpt, dim_cols=dim_cols)
        x = x_full[landmark_idx]

        c = make_cost(x) / scale

        for template_name, c_template_raw in template_costs.items():
            c_template = c_template_raw / scale

            dist = entropic_gw_distance(
                c,
                c_template,
                args.epsilon,
                args.max_iter,
                args.tol,
            )

            rows.append(
                {
                    "checkpoint_path": str(ckpt),
                    "epoch": int(epoch_from_path(ckpt)),
                    "template": template_name,
                    "gw_to_template": float(dist),
                    "n_landmarks": int(len(landmark_idx)),
                    "landmark_seed": int(args.seed),
                    "epsilon": float(args.epsilon),
                    "max_iter": int(args.max_iter),
                    "tol": float(args.tol),
                    "low_rank": int(args.low_rank),
                    "normalization": args.normalize,
                    "normalization_common_scale": float(scale),
                }
            )

    out = pd.DataFrame(rows)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix == ".parquet":
        out.to_parquet(output, index=False)
    elif output.suffix == ".csv":
        out.to_csv(output, index=False)
    else:
        raise ValueError("Output must end in .parquet or .csv")

    print(f"[DONE] wrote {output}")


if __name__ == "__main__":
    main()
