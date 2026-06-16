""" Synthetic geometry samplers for collapse experiments. """
import numpy as np


def isotropic_init(_n, _d, _seed=None):
    """
    Isotropic point clouds.

    :param _n: num points.
    :param _d: ambient dimension.
    :param _seed: baseline random seed.
    :return: numpy point cloud.
    """
    r_num = np.random.default_rng(_seed)
    x = r_num.normal(size=(_n, _d))
    x = (x - x.mean(axis=0, keepdims=True)) / (
        x.std(axis=0, keepdims=True) + 1e-8
    )
    return x


def kcube_init(_n, _d, _k, _seed=None):
    """
    k-cube point clouds.

    :param _n: num points.
    :param _d: ambient dimension.
    :param _k: down proj dim.
    :param _seed: baseline random seed.
    :return: numpy point cloud.
    """
    r_num = np.random.default_rng(_seed)
    x = np.zeros((_n, _d))
    x[:, :_k] = r_num.uniform(-1.0, 1.0, size=(_n, _k))
    return x


def kplane_init(_n, _d, _k, _seed=None):
    """
    k-plane point clouds.

    :param _n: num points.
    :param _d: ambient dimension.
    :param _k: down proj dim.
    :param _seed: baseline random seed.
    :return: numpy point cloud.
    """
    r_num = np.random.default_rng(_seed)
    x = np.zeros((_n, _d))
    x[:, :_k] = r_num.normal(size=(_n, _k))
    return x


def sphere_init(_n, _d, _r=1.0, _seed=None):
    """
    Sphere point clouds.

    :param _n: num points.
    :param _d: ambient dimension.
    :param _r: radius.
    :param _seed: baseline random seed.
    :return: numpy point cloud.
    """
    r_num = np.random.default_rng(_seed)
    x = r_num.normal(size=(_n, _d))
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    x *= _r
    return x


def torus_init(_n, _d, _r_major=2.0, _r_minor=0.5, _seed=None):
    """
    Torus point clouds.

    :param _n: num points.
    :param _d: ambient dimension.
    :param _r_major: major radius.
    :param _r_minor: minor radius.
    :param _seed: baseline random seed.
    :return: numpy point cloud.
    """
    r_num = np.random.default_rng(_seed)
    theta = r_num.uniform(0.0, 2.0 * np.pi, size=_n)
    phi = r_num.uniform(0.0, 2.0 * np.pi, size=_n)

    x = np.zeros((_n, _d))
    x[:, 0] = (_r_major + _r_minor * np.cos(phi)) * np.cos(theta)
    x[:, 1] = (_r_major + _r_minor * np.cos(phi)) * np.sin(theta)
    x[:, 2] = _r_minor * np.sin(phi)
    return x


def swiss_init(_n, _d, _seed=None):
    """
    Swiss roll point cloud.

    :param _n: num points.
    :param _d: ambient dimension.
    :param _seed: baseline random seed.
    :return: numpy point cloud.
    """
    r_num = np.random.default_rng(_seed)
    t = r_num.uniform(1.5 * np.pi, 4.5 * np.pi, size=_n)
    h = r_num.uniform(-1.0, 1.0, size=_n)

    x = np.zeros((_n, _d))
    x[:, 0] = t * np.cos(t)
    x[:, 1] = h
    x[:, 2] = t * np.sin(t)
    return x


def paraboloid_init(_n, _d, _a=1.0, _seed=None):
    """
    Paraboloid point cloud.

    :param _n: num points.
    :param _d: ambient dimension.
    :param _a:
    :param _seed: baseline random seed.
    :return: numpy point cloud.
    """
    r_num = np.random.default_rng(_seed)
    u = r_num.uniform(-1.0, 1.0, size=(_n, _d - 1))
    z = _a * np.sum(u ** 2, axis=1, keepdims=True)
    x = np.concatenate([u, z], axis=1)
    return x


def spiked_gaussian_init(_n, _d, _spike=5.0, _seed=None):
    """
    Spiked Gaussian point cloud.

    :param _n: num points.
    :param _d: ambient dimension.
    :param _spike: spike multiple factor.
    :param _seed: baseline random seed.
    :return: numpy point cloud.
    """
    r_num = np.random.default_rng(_seed)
    x = r_num.normal(size=(_n, _d))
    x[:, 0] *= _spike
    return x

def ring_init(_n, _d, _r=1.0, _width=0.15, _seed=None):
    """
    Ring-like point cloud.

    :param _n: num points.
    :param _d: ambient dimension.
    :param _r: ring radius.
    :param _width: width factor of ring.
    :param _seed: baseline random seed.
    :return: numpy point cloud.
    """
    r_num = np.random.default_rng(_seed)
    theta = r_num.uniform(0.0, 2.0 * np.pi, size=_n)
    rad = _r + r_num.uniform(-_width, _width, size=_n)

    x = np.zeros((_n ,_d))
    x[:, 0] = rad * np.cos(theta)
    x[:, 1] = rad * np.sin(theta)
    return x


def make_clustered_gaussian(_n, _d, _num_clusters=4, _cluster_std=0.5,
    _center_scale=4.0, _seed=0, _shuffle=True, _return_centers=False):
    """
    Clustered Gaussian, one of the most relevant point clouds.

    :param _n: num points.
    :param _d: ambient dimension.
    :param _num_clusters: initial number of clusters.
    :param _cluster_std: standard deviation of each cluster.
    :param _center_scale: scale for each center.
    :param _seed: baseline random seed.
    :param _shuffle: shuffle around the clusters based on random seed.
    :param _return_centers: bool, return centers in needed.
    :return: numpy point cloud, list of labels.
    """
    if _n <= 0:
        raise ValueError(f"n must be positive, got {_n}")
    if _d <= 0:
        raise ValueError(f"d must be positive, got {_d}")
    if _num_clusters <= 0:
        raise ValueError(f"num_clusters must be positive, got {_num_clusters}")

    rng = np.random.default_rng(_seed)

    counts = np.full(_num_clusters, _n // _num_clusters, dtype=int)
    counts[: _n % _num_clusters] += 1

    centers = rng.normal(loc=0.0, scale=_center_scale, size=(_num_clusters, _d))

    xs = []
    ys = []
    for k, nk in enumerate(counts):
        pts = centers[k] + rng.normal(loc=0.0, scale=_cluster_std, size=(nk, _d))
        labs = np.full(nk, k, dtype=int)
        xs.append(pts)
        ys.append(labs)

    x = np.vstack(xs)
    labels = np.concatenate(ys)

    if _shuffle:
        perm = rng.permutation(_n)
        x = x[perm]
        labels = labels[perm]

    if _return_centers:
        return x, labels, centers
    return x, labels


def get_geometry(_name, _n, _d, _seed=None, _k=2):
    """
    Fetch each baseline geometry from the name, points, dim, seed, and k proj dim.

    :param _name: string name for the cloud.
    :param _n: num points.
    :param _d: ambient dimension.
    :param _seed: baseline random seed.
    :param _k: optional down-proj dim.
    :return: point cloud per call.
    """
    if _name == "isotropic":
        return isotropic_init(_n, _d, _seed)
    if _name == "kcube":
        return kcube_init(_n, _d, _k, _seed)
    if _name == "kplane":
        return kplane_init(_n, _d, _k, _seed)
    if _name == "sphere":
        return sphere_init(_n, _d, _seed=_seed)
    if _name == "torus":
        return torus_init(_n, _d, _seed=_seed)
    if _name == "swiss":
        return swiss_init(_n, _d, _seed)
    if _name == "paraboloid":
        return paraboloid_init(_n, _d, _seed=_seed)
    if _name == "spiked_gaussian":
        return spiked_gaussian_init(_n, _d, _seed=_seed)
    if _name == "ring":
        return ring_init(_n, _d, _seed=_seed)
    if _name == "clustered_gaussian":
        return make_clustered_gaussian(_n, _d, _seed=_seed)

    raise ValueError(f"Unknown geometry: {_name}")
