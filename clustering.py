""" Clustering label helpers when required for particular geometries. """
import argparse
import numpy as np


def make_clustered_gaussian(
        n: int,
        d: int,
        num_clusters: int = 4,
        cluster_std: float = 0.5,
        center_scale: float = 4.0,
        seed: int = 0,
        shuffle: bool = True,
        return_centers: bool = False,
):
    """
    Generate a Gaussian mixture / clustered Gaussian point cloud.

    :param n: number of points.
    :param d: ambient dimension.
    :param num_clusters: number of clusters.
    :param cluster_std: standard deviation within each cluster.
    :param center_scale: scale used to sample cluster centers.
    :param seed: random seed.
    :param shuffle: bool, shuffle points after construction.
    :param return_centers: also return the cluster centers.

    :returns:
    - x: sampled point cloud.
    - labels: integer cluster labels.
    - centers: cluster centers if return_centers=True.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")
    if num_clusters <= 0:
        raise ValueError(f"num_clusters must be positive, got {num_clusters}")

    rng = np.random.default_rng(seed)

    counts = np.full(num_clusters, n // num_clusters, dtype=int)
    counts[: n % num_clusters] += 1

    centers = rng.normal(loc=0.0, scale=center_scale, size=(num_clusters, d))

    xs = []
    ys = []

    for k, nk in enumerate(counts):
        xk = centers[k] + rng.normal(loc=0.0, scale=cluster_std, size=(nk, d))
        yk = np.full(nk, k, dtype=int)
        xs.append(xk)
        ys.append(yk)

    x = np.vstack(xs)
    labels = np.concatenate(ys)

    if shuffle:
        perm = rng.permutation(n)
        x = x[perm]
        labels = labels[perm]

    if return_centers:
        return x, labels, centers
    return x, labels


def get_cluster_labels_for_geometry(_exp, _x0):
    """
    Return integer cluster labels for use with cluster-based collapse mechanisms.

    Strategy:
    1. If the experiment already carries labels, use them.
    2. If the geometry is explicitly clustered, synthesize deterministic labels.
    3. Optionally infer labels from the point cloud for a few known cases.
    4. Otherwise fail.

    :param _exp: experiment cluster labels.
    :param: _x0: point cloud.

    :return: labels: integer cluster assignments in {0, 1, ..., K-1}.
    """
    n = len(_x0)

    # 1. Explicit labels win if already attached to the experiment.
    if hasattr(_exp, "cluster_labels") and _exp.cluster_labels is not None:
        labels = np.asarray(_exp.cluster_labels)
        if labels.shape[0] != n:
            raise ValueError(
                f"_exp.cluster_labels has length {labels.shape[0]}, expected {n}"
            )
        return labels.astype(int)

    geom = getattr(_exp, "base_geometry", None)
    if geom is None:
        raise ValueError("Experiment has no base_geometry")

    # Optional override for number of clusters.
    k_clusters = getattr(_exp, "num_clusters", None)
    if k_clusters is None:
        k_clusters = getattr(_exp, "cluster_k", None)
    if k_clusters is None:
        k_clusters = 4

    if geom in {
        "clustered_gaussian",
        "gaussian_mixture",
        "cluster_mixture",
        "prototype_cloud",
    }:
        return _balanced_block_labels(n, k_clusters)

    if geom == "kcube":
        m = min(_x0.shape[1], max(1, int(np.ceil(np.log2(k_clusters)))))
        bits = (_x0[:, :m] > 0).astype(int)
        raw = bits.dot(1 << np.arange(m))
        return _compress_labels(raw, k_clusters)

    if geom == "spiked_gaussian":
        m = min(_x0.shape[1], max(1, int(np.ceil(np.log2(k_clusters)))))
        bits = (_x0[:, :m] > np.median(_x0[:, :m], axis=0)).astype(int)
        raw = bits.dot(1 << np.arange(m))
        return _compress_labels(raw, k_clusters)

    if getattr(_exp, "allow_inferred_clusters", False):
        return _simple_kmeans_labels(
            _x0,
            n_clusters=k_clusters,
            seed=getattr(_exp, "seed", 0),
            n_iter=25,
        )

    raise NotImplementedError(
        f"Geometry '{geom}' does not have built-in cluster labels. "
        "Either attach _exp.cluster_labels, use a clustered geometry, "
        "or set allow_inferred_clusters=True for a simple k-means fallback."
    )


def _balanced_block_labels(n, k):
    """
    Deterministic labels [0,0,...,1,1,...,k-1] with nearly equal block sizes.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    counts = np.full(k, n // k, dtype=int)
    counts[: n % k] += 1
    labels = np.concatenate([np.full(c, i, dtype=int) for i, c in enumerate(counts)])
    return labels


def _compress_labels(raw_labels, k):
    """
    Map arbitrary nonnegative integer labels into {0, ..., k-1}.
    If there are more than k unique raw labels, fold them modulo k.
    """
    raw_labels = np.asarray(raw_labels, dtype=int)
    uniq = np.unique(raw_labels)

    if len(uniq) <= k:
        lut = {u: i for i, u in enumerate(uniq)}
        return np.array([lut[u] for u in raw_labels], dtype=int)

    return (raw_labels % k).astype(int)


def _simple_kmeans_labels(x, n_clusters, seed=0, n_iter=25):
    """
    Small dependency-light k-means fallback.
    Not meant for publication-quality clustering, only for wiring the mechanism.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if n_clusters <= 0:
        raise ValueError(f"n_clusters must be positive, got {n_clusters}")
    if n_clusters > n:
        raise ValueError(f"n_clusters={n_clusters} exceeds number of points n={n}")

    rng = np.random.default_rng(seed)
    init_idx = rng.choice(n, size=n_clusters, replace=False)
    centers = x[init_idx].copy()

    labels = np.zeros(n, dtype=int)

    for _ in range(n_iter):
        dists = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dists, axis=1)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for j in range(n_clusters):
            mask = labels == j
            if mask.any():
                centers[j] = x[mask].mean(axis=0)
            else:
                # Re-seed empty cluster
                centers[j] = x[rng.integers(0, n)]

    return labels


def test(_args):
    """
    Dumb little functionality test.
    :param _args: point cloud/cluster num args.
    :return: just print to check.
    """
    x, labs1 = make_clustered_gaussian(_args.n, _args.d, _args.k)
    labs2 = _simple_kmeans_labels(x, _args.k)
    labs3 = _balanced_block_labels(_args.n, _args.k)
    print('=== Labs 1 (OPC) ===')
    print(labs1)
    print('=== Labs 2 (KNN) ===')
    print(labs2)
    print('=== Labs 3 (BLO) ===')
    print(labs3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cluster helpers')
    parser.add_argument("-d", type=int)
    parser.add_argument("-n", type=int)
    parser.add_argument("-k", type=int)
    args = parser.parse_args()
    test(args)
