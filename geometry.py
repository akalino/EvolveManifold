""" Synthetic geometry samplers for collapse experiments. """
import numpy as np


def isotropic_init(_n, _d, _seed=None):
    """

    :param _n:
    :param _d:
    :param _seed:
    :return:
    """
    r_num = np.random.default_rng(_seed)
    x = r_num.normal(size=(_n, _d))
    x = (x - x.mean(axis=0, keepdims=True)) / (
        x.std(axis=0, keepdims=True) + 1e-8
    )
    return x


def kcube_init(_n, _d, _k, _seed=None):
    """

    :param _n:
    :param _d:
    :param _k:
    :param _seed:
    :return:
    """
    r_num = np.random.default_rng(_seed)
    x = np.zeros((_n, _d))
    x[:, :_k] = r_num.uniform(-1.0, 1.0, size=(_n, _k))
    return x


def kplane_init(_n, _d, _k, _seed=None):
    """

    :param _n:
    :param _d:
    :param _k:
    :param _seed:
    :return:
    """
    r_num = np.random.default_rng(_seed)
    x = np.zeros((_n, _d))
    x[:, :_k] = r_num.normal(size=(_n, _k))
    return x


def sphere_init(_n, _d, _r=1.0, _seed=None):
    """

    :param _n:
    :param _d:
    :param _r:
    :param _seed:
    :return:
    """
    r_num = np.random.default_rng(_seed)
    x = r_num.normal(size=(_n, _d))
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    x *= _r
    return x


def torus_init(_n, _d, _r_major=2.0, _r_minor=0.5, _seed=None):
    """

    :param _n:
    :param _d:
    :param _r_major:
    :param _r_minor:
    :param _seed:
    :return:
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

    :param _n:
    :param _d:
    :param _seed:
    :return:
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

    :param _n:
    :param _d:
    :param _a:
    :param _seed:
    :return:
    """
    r_num = np.random.default_rng(_seed)
    u = r_num.uniform(-1.0, 1.0, size=(_n, _d - 1))
    z = _a * np.sum(u ** 2, axis=1, keepdims=True)
    x = np.concatenate([u, z], axis=1)
    return x


def spiked_gaussian_init(_n, _d, _spike=5.0, _seed=None):
    """

    :param _n:
    :param _d:
    :param _spike:
    :param _seed:
    :return:
    """
    r_num = np.random.default_rng(_seed)
    x = r_num.normal(size=(_n, _d))
    x[:, 0] *= _spike
    return x

def ring_init(_n, _d, _r=1.0, _width=0.15, _seed=None):
    """
    :param _n:
    :param _d:
    :param _r:
    :param _width:
    :param _seed:
    """
    r_num = np.random.default_rng(_seed)
    theta = r_num.uniform(0.0, 2.0 * np.pi, size=_n)
    rad = _r + r_num.uniform(-_width, _width, size=_n)

    x = np.zeros((_n ,_d))
    x[:, 0] = rad * np.cos(theta)
    x[:, 1] = rad * np.sin(theta)
    return x


def get_geometry(_name, _n, _d, _seed=None, _k=2):
    """

    :param _name:
    :param _n:
    :param _d:
    :param _seed:
    :param _k:
    :return:
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

    raise ValueError(f"Unknown geometry: {_name}")
