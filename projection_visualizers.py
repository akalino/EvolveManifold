"""
projection_visualizers.py

Reusable 2D projection helpers for visualizing EvolveManifold trajectories.

Primary entry point:

    zs, subtitle = project_joint_snapshots(
        xs=[x0, x_mid, x_final],
        method="pca",
        labels=labels,
        seed=0,
    )

where:
    xs is a list of point clouds with shape (n, d)
    zs is a list of projected arrays with shape (n, 2)

The projection is fit jointly across all selected checkpoints whenever
appropriate, so early/middle/late panels live in a shared coordinate system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler


try:
    import umap  # type: ignore
except ImportError:  # pragma: no cover
    umap = None


Array = np.ndarray


@dataclass(frozen=True)
class ProjectionResult:
    zs: list[Array]
    subtitle: str
    method: str


def _validate_snapshots(xs: list[Array]) -> list[Array]:
    if not xs:
        raise ValueError("xs must contain at least one snapshot.")

    out = [np.asarray(x) for x in xs]

    n0 = out[0].shape[0]
    d0 = out[0].shape[1]

    for i, x in enumerate(out):
        if x.ndim != 2:
            raise ValueError(f"snapshot {i} must be 2D, got shape {x.shape}")
        if x.shape[0] != n0:
            raise ValueError(
                "all snapshots must have the same number of points; "
                f"snapshot 0 has {n0}, snapshot {i} has {x.shape[0]}"
            )
        if x.shape[1] != d0:
            raise ValueError(
                "all snapshots must have the same ambient dimension; "
                f"snapshot 0 has {d0}, snapshot {i} has {x.shape[1]}"
            )

    return out


def _split_joint_projection(z_all: Array, xs: list[Array]) -> list[Array]:
    sizes = [x.shape[0] for x in xs]
    splits = np.cumsum(sizes)[:-1]
    return list(np.split(z_all, splits))


def _joint_array(xs: list[Array], standardize: bool = False) -> Array:
    x_all = np.vstack(xs)
    if standardize:
        x_all = StandardScaler().fit_transform(x_all)
    return x_all


def project_raw_coordinates(
    xs: list[Array],
    coord_a: int = 0,
    coord_b: int = 1,
) -> ProjectionResult:
    xs = _validate_snapshots(xs)
    d = xs[0].shape[1]

    if coord_a >= d or coord_b >= d:
        raise ValueError(
            f"requested coordinates ({coord_a}, {coord_b}) but dimension is {d}"
        )

    zs = [x[:, [coord_a, coord_b]] for x in xs]
    return ProjectionResult(
        zs=zs,
        subtitle=f"Raw coordinates {coord_a},{coord_b}",
        method=f"raw{coord_a}{coord_b}",
    )


def project_pca(
    xs: list[Array],
    seed: int = 0,
    standardize: bool = False,
) -> ProjectionResult:
    xs = _validate_snapshots(xs)
    x_all = _joint_array(xs, standardize=standardize)

    pca = PCA(n_components=2, random_state=seed)
    z_all = pca.fit_transform(x_all)
    zs = _split_joint_projection(z_all, xs)

    evr = pca.explained_variance_ratio_
    label = "Standardized PCA" if standardize else "PCA"

    return ProjectionResult(
        zs=zs,
        subtitle=f"{label}: PC1={evr[0]:.3f}, PC2={evr[1]:.3f}",
        method="pca_standardized" if standardize else "pca",
    )


def project_random(
    xs: list[Array],
    seed: int = 0,
    standardize: bool = False,
) -> ProjectionResult:
    xs = _validate_snapshots(xs)
    x_all = _joint_array(xs, standardize=standardize)

    rp = GaussianRandomProjection(n_components=2, random_state=seed)
    z_all = rp.fit_transform(x_all)
    zs = _split_joint_projection(z_all, xs)

    label = "Standardized Gaussian random projection" if standardize else "Gaussian random projection"

    return ProjectionResult(
        zs=zs,
        subtitle=label,
        method="random_standardized" if standardize else "random",
    )


def project_umap(
    xs: list[Array],
    seed: int = 0,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    standardize: bool = False,
) -> ProjectionResult:
    if umap is None:
        raise ImportError(
            "umap-learn is not installed. Install with: pip install umap-learn"
        )

    xs = _validate_snapshots(xs)
    x_all = _joint_array(xs, standardize=standardize)

    reducer = umap.UMAP(
        n_components=2,
        random_state=seed,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
    )
    z_all = reducer.fit_transform(x_all)
    zs = _split_joint_projection(z_all, xs)

    label = (
        f"UMAP: neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}"
    )
    if standardize:
        label = "Standardized " + label

    return ProjectionResult(
        zs=zs,
        subtitle=label,
        method="umap_standardized" if standardize else "umap",
    )


def project_tsne(
    xs: list[Array],
    seed: int = 0,
    perplexity: float = 30.0,
    standardize: bool = False,
) -> ProjectionResult:
    """
    Exploratory only. t-SNE can make attractive panels, but it is usually less
    defensible than raw coordinates, PCA, or mechanism-aware projections for a
    benchmark paper.
    """
    xs = _validate_snapshots(xs)
    x_all = _joint_array(xs, standardize=standardize)

    reducer = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )
    z_all = reducer.fit_transform(x_all)
    zs = _split_joint_projection(z_all, xs)

    label = f"t-SNE: perplexity={perplexity}"
    if standardize:
        label = "Standardized " + label

    return ProjectionResult(
        zs=zs,
        subtitle=label,
        method="tsne_standardized" if standardize else "tsne",
    )


def project_radial_pc1(
    xs: list[Array],
    seed: int = 0,
    center: str = "global_mean",
) -> ProjectionResult:
    """
    Mechanism-aware projection useful for radial collapse.

    x-axis: first joint PC
    y-axis: distance from a fixed center

    The center is computed jointly across selected checkpoints by default.
    """
    xs = _validate_snapshots(xs)
    x_all = np.vstack(xs)

    if center == "global_mean":
        c = x_all.mean(axis=0, keepdims=True)
    elif center == "origin":
        c = np.zeros((1, x_all.shape[1]))
    else:
        raise ValueError("center must be 'global_mean' or 'origin'")

    pca = PCA(n_components=1, random_state=seed)
    pc1_all = pca.fit_transform(x_all)[:, 0]
    radius_all = np.linalg.norm(x_all - c, axis=1)

    z_all = np.column_stack([pc1_all, radius_all])
    zs = _split_joint_projection(z_all, xs)

    return ProjectionResult(
        zs=zs,
        subtitle=f"Mechanism-aware: PC1 vs radius ({center})",
        method=f"radial_pc1_{center}",
    )


def project_norm_time(
    xs: list[Array],
    center: str = "global_mean",
) -> ProjectionResult:
    """
    Diagnostic projection useful when scatter geometry is visually confusing.

    x-axis: checkpoint index repeated for each point
    y-axis: point radius/norm from a fixed center

    This does not show 2D shape, but it clearly shows contraction/expansion.
    """
    xs = _validate_snapshots(xs)
    x_all = np.vstack(xs)

    if center == "global_mean":
        c = x_all.mean(axis=0, keepdims=True)
    elif center == "origin":
        c = np.zeros((1, x_all.shape[1]))
    else:
        raise ValueError("center must be 'global_mean' or 'origin'")

    zs = []
    for t, x in enumerate(xs):
        radius = np.linalg.norm(x - c, axis=1)
        time_coord = np.full_like(radius, fill_value=float(t), dtype=float)
        zs.append(np.column_stack([time_coord, radius]))

    return ProjectionResult(
        zs=zs,
        subtitle=f"Diagnostic: checkpoint index vs radius ({center})",
        method=f"norm_time_{center}",
    )


def project_cluster_centroids(
    xs: list[Array],
    labels: Array,
    seed: int = 0,
) -> ProjectionResult:
    """
    Mechanism-aware projection for cluster merging/tightening.

    Computes cluster centroids at each checkpoint, fits PCA to all centroids
    jointly, then returns projected centroids rather than projected points.

    This is not a drop-in replacement for point-level scatter unless the
    plotting code can handle a smaller number of points per snapshot.
    """
    xs = _validate_snapshots(xs)
    labels = np.asarray(labels)

    if labels.ndim != 1 or labels.shape[0] != xs[0].shape[0]:
        raise ValueError("labels must be a 1D array with one label per point")

    unique_labels = np.unique(labels)
    centroid_snapshots = []

    for x in xs:
        centroids = []
        for lab in unique_labels:
            centroids.append(x[labels == lab].mean(axis=0))
        centroid_snapshots.append(np.vstack(centroids))

    centroids_all = np.vstack(centroid_snapshots)
    pca = PCA(n_components=2, random_state=seed)
    z_all = pca.fit_transform(centroids_all)
    zs = _split_joint_projection(z_all, centroid_snapshots)

    evr = pca.explained_variance_ratio_

    return ProjectionResult(
        zs=zs,
        subtitle=f"Cluster centroids PCA: PC1={evr[0]:.3f}, PC2={evr[1]:.3f}",
        method="cluster_centroids_pca",
    )


def project_joint_snapshots(
    xs: list[Array],
    method: str = "pca",
    labels: Array | None = None,
    seed: int = 0,
    standardize: bool = False,
    **kwargs,
) -> ProjectionResult:
    """
    Main dispatcher.

    Supported methods:
      raw01
      raw02
      raw12
      pca
      pca_standardized
      random
      random_standardized
      umap
      umap_standardized
      tsne
      tsne_standardized
      radial
      radial_origin
      norm_time
      norm_time_origin
      cluster_centroids
    """
    method = method.lower()

    if method == "raw01":
        return project_raw_coordinates(xs, 0, 1)

    if method == "raw02":
        return project_raw_coordinates(xs, 0, 2)

    if method == "raw12":
        return project_raw_coordinates(xs, 1, 2)

    if method == "pca":
        return project_pca(xs, seed=seed, standardize=standardize)

    if method == "pca_standardized":
        return project_pca(xs, seed=seed, standardize=True)

    if method == "random":
        return project_random(xs, seed=seed, standardize=standardize)

    if method == "random_standardized":
        return project_random(xs, seed=seed, standardize=True)

    if method == "umap":
        return project_umap(xs, seed=seed, standardize=standardize, **kwargs)

    if method == "umap_standardized":
        return project_umap(xs, seed=seed, standardize=True, **kwargs)

    if method == "tsne":
        return project_tsne(xs, seed=seed, standardize=standardize, **kwargs)

    if method == "tsne_standardized":
        return project_tsne(xs, seed=seed, standardize=True, **kwargs)

    if method == "radial":
        return project_radial_pc1(xs, seed=seed, center="global_mean")

    if method == "radial_origin":
        return project_radial_pc1(xs, seed=seed, center="origin")

    if method == "norm_time":
        return project_norm_time(xs, center="global_mean")

    if method == "norm_time_origin":
        return project_norm_time(xs, center="origin")

    if method == "cluster_centroids":
        if labels is None:
            raise ValueError("cluster_centroids projection requires labels")
        return project_cluster_centroids(xs, labels=labels, seed=seed)

    raise ValueError(f"Unknown projection method: {method}")


AVAILABLE_PROJECTIONS = [
    "raw01",
    "raw02",
    "raw12",
    "pca",
    "pca_standardized",
    "random",
    "random_standardized",
    "umap",
    "umap_standardized",
    "tsne",
    "tsne_standardized",
    "radial",
    "radial_origin",
    "norm_time",
    "norm_time_origin",
    "cluster_centroids",
]