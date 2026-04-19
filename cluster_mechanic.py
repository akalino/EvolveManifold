""" Cluster-structure collapse mechanisms."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from schedulers import get_schedule_value


Array = np.ndarray

@dataclass
class ClusterParams:
    """Dataclass to hold cluster collapse parameters."""
    schedule: str
    start_strength: float
    end_strength: float
    finish: int
    cluster_labels: Array
    mover_frac: float = 1.0
    num_clusters: int = 1
    mode: str = "tighten"  # supports tighten, merge
    merge_target: str = "global"  # supports global, pairwise
    merge_pairs: Optional[Sequence[tuple]] = None
    fixed_indices: Optional[Array] = None
    seed: int = 0


def _choose_indices(_n, _mover_frac, _rng):
    """
    Chooses the indices of points moved.
    :param _n: Overall number of points.
    :param _mover_frac: Fraction of points moved.
    :param _rng: Random generator.
    :return: List of point indices.
    """
    m = max(1, int(round(_mover_frac * _n)))
    return _rng.choice(_n, size=m, replace=False)


def _cluster_centroids(_x, _labels):
    """
    Cluster centroids of each cluster.
    :param _x: Input point cloud.
    :param _labels: Array of labels for centroids.
    :return: D     ict of centroids.
    """
    centroids = {}
    for c in np.unique(_labels):
        mask = _labels == c
        centroids[int(c)] = _x[mask].mean(axis=0)
    return centroids


def step_cluster_collapse(p: ClusterParams):
    rng = np.random.default_rng(p.seed)
    labels = np.asarray(p.cluster_labels)

    def _step(_x, _t, _rng) -> Array:
        x = np.asarray(_x)
        xn = x.copy()

        lam = get_schedule_value(
            p.schedule,
            p.start_strength,
            p.end_strength,
            p.finish,
            _t
        )

        if len(labels) != len(x):
            raise ValueError("cluster_labels must have same length as x")

        if p.fixed_indices is None:
            move_idx = _choose_indices(len(x), p.mover_frac, _rng)
        else:
            move_idx = p.fixed_indices

        centroids = _cluster_centroids(x, labels)

        if p.mode == "tighten":
            for i in move_idx:
                c = centroids[int(labels[i])]
                xn[i] = (1.0 - lam) * x[i] + lam * c

        elif p.mode == "merge":
            global_center = x.mean(axis=0)

            moved_centroids = {}
            if p.merge_target == "global":
                for c, mu in centroids.items():
                    moved_centroids[c] = (1.0 - lam) * mu + lam * global_center
            elif p.merge_target == "pairwise":
                if p.merge_pairs is None:
                    raise ValueError("merge_pairs required for pairwise merge")
                moved_centroids = dict(centroids)
                for a, b in p.merge_pairs:
                    mid = 0.5 * (centroids[a] + centroids[b])
                    moved_centroids[a] = (1.0 - lam) * centroids[a] + lam * mid
                    moved_centroids[b] = (1.0 - lam) * centroids[b] + lam * mid
            else:
                raise ValueError(f"Unknown merge_target: {p.merge_target}")

            for i in move_idx:
                c = int(labels[i])
                delta = moved_centroids[c] - centroids[c]
                xn[i] = x[i] + delta

        else:
            raise ValueError(f"Unknown cluster collapse mode: {p.mode}")

        return xn

    return _step


def cluster_params_from_severity(
    severity: str,
    schedule: str,
    finish: int,
    cluster_labels: Array,
    mover_frac: float = 1.0,
    mode: str = "tighten",
    seed: int = 0,
) -> ClusterParams:
    if severity == "weak":
        start_strength, end_strength = 0.0, 0.02
    elif severity == "moderate":
        start_strength, end_strength = 0.0, 0.05
    elif severity == "strong":
        start_strength, end_strength = 0.0, 0.10
    else:
        raise ValueError(f"Unknown severity: {severity}")

    return ClusterParams(
        schedule=schedule,
        start_strength=start_strength,
        end_strength=end_strength,
        finish=finish,
        cluster_labels=cluster_labels,
        mover_frac=mover_frac,
        mode=mode,
        seed=seed
    )