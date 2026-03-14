""" Various manifold projections for collapse. """
import numpy as np


def proj_to_sphere(_x, _r, _eps=1e-12):
    """

    :param _x: Point cloud.
    :param _r: Radius of sphere.
    :param _eps: Numerical stability.
    :return:
    """
    norms = np.linalg.norm(_x, axis=1, keepdims=True)
    return (_r / (norms + _eps)) * _x


def proj_to_k_plane(_x, _k):
    """

    :param _x: Point cloud.
    :param _k: Sub-k dimension.
    :return:
    """
    y = _x.copy()
    if not 0 <= _k <= y.shape[1]:
        raise ValueError("Need 0 <= _k <= y.shape[1]")
    y[:, _k:] = 0.0
    return y


def proj_to_swiss(_x):
    """

    :param _x: Point cloud.
    :return:
    """
    y = _x.copy()
    if _x.shape[1] < 3:
        raise ValueError("Need d>=3 for Swiss projection")
    x0, x1, x2 = _x[:, 0], _x[:, 1], _x[:, 2]
    u = np.sqrt(x0 ** 2 + x2 ** 2)  # approximate inverse
    y[:, 0] = u * np.cos(u)
    y[:, 2] = u * np.sin(u)
    y[:, 1] = x1
    return y


def proj_to_torus(_x, _r_major=2.0, _r_minor=0.5, _dims=(0, 1, 2), _eps=1e-8):
    """

    :param _x: Point cloud.
    :param _r_major: Major radius.
    :param _r_minor: Minor radius.
    :param _dims: Dimensions.
    :param _eps: Numerical stability.
    """
    _y = np.asarray(_x, dtype=float).copy()
    ix, iy, iz = _dims

    x = _y[:, ix]
    y = _y[:, iy]
    z = _y[:, iz]

    rho = np.sqrt(x ** 2 + y ** 2)
    ux = x / (rho + _eps)
    uy = y / (rho + _eps)
    cx = _r_major * ux
    cy = _r_major * uy
    cz = 0.0

    vx = x - cx
    vy = y - cy
    vz = z - cz
    v_norm = np.sqrt(vx * vx + vy * vy + vz * vz) + _eps
    _y[:, ix] = cx + _r_minor * (vx / v_norm)
    _y[:, iy] = cy + _r_minor * (vy / v_norm)
    _y[:, iz] = cz + _r_minor * (vz / v_norm)
    return _y


def proj_to_paraboloid(_x, _a=1.0, _u_dims=None, _z_dims=None,
                       _iters=25, _tol=1e-10, _eps=1e-8):
    """

    :param _x:
    :param _a:
    :param _u_dims:
    :param _z_dims:
    :param _iters:
    :param _tol:
    :param _eps:
    """
    _x = np.asarray(_x, dtype=float)
    _, d = _x.shape
    _y = _x.copy()

    if _u_dims is None:
        _u_dims = slice(0, d - 1)
    if _z_dims is None:
        _z_dims = d - 1


    u = _y[:, _u_dims]
    z = _y[:, _z_dims].copy()

    s = np.linalg.norm(u, axis=1)  # ||u||
    t = np.maximum(s, 0.0)

    # Newton solve
    az = _a * z
    c1 = 1.0 - 2.0 * az

    for _ in range(_iters):
        f = 2.0 * (_a * _a) * (t ** 3) + c1 * t - s
        fp = 6.0 * (_a * _a) * (t ** 2) + c1

        step = f / (fp + _eps)
        t_new = t - step

        # keep t nonnegative (radius)
        t_new = np.maximum(t_new, 0.0)

        if np.max(np.abs(t_new - t)) < _tol:
            t = t_new
            break
        t = t_new

    scale = np.where(s > _eps, t / (s + _eps), 0.0)  # (n,)
    u_proj = u * scale[:, None]
    z_proj = _a * (t ** 2)

    _y[:, _u_dims] = u_proj
    _y[:, _z_dims] = z_proj
    return _y


def proj_to_k_cube(_x, _k, _low=0.0, _high=1.0, _dims=None, _zero=False):
    """
    :param _x:
    :param _k:
    :param _low:
    :param _high:
    :param _dims:
    :param _zero:
    """
    x = np.asarray(_x, dtype=float)
    _, d = _x.shape
    if not 0 <= _k <= d:
        raise ValueError("k must satisfy 0 <= k <= d")

    y = x.copy()
    if _dims is None:
        _dims = slice(0, _k)

    y[:, _dims] = np.clip(y[:, _dims], _low, _high)

    if _zero:
        mask = np.ones(d, dtype=bool)
        if isinstance(_dims, slice):
            idx = np.arange(d)[_dims]
        else:
            idx = np.array(_dims, dtype=int)
        mask[idx] = False
        y[:, mask] = 0.0

    return y
