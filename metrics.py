"""
Geometric, spectral, and topological metric computation functions.
TODO: this could be a good spot to put mixup barcodes in the future.
"""
import numpy as np


def effective_rank(x, eps=1e-12):
    """
    Computes effective rank.

    :param x: point cloud.
    :param eps: num. rounding.
    :return: eff rank.
    """
    xc = x - x.mean(axis=0, keepdims=True)
    cov = (xc.T @ xc) / xc.shape[0]
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0.0, None)

    total = eigvals.sum()
    if total <= eps:
        return 0.0

    p = eigvals / total
    p = p[p > eps]
    return float(np.exp(-(p * np.log(p)).sum()))


def top_k_variance_fraction(x, k):
    """
    Frac of top_k variance to total variance.

    :param x: point cloud.
    :param k: dim to consider.
    :return: top_k variance.
    """
    xc = x - x.mean(axis=0, keepdims=True)
    cov = (xc.T @ xc) / xc.shape[0]
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0.0, None)[::-1]

    total = eigvals.sum()
    if total <= 0:
        return 0.0

    return float(eigvals[:k].sum() / total)


def mean_pairwise_distance(x):
    """
    Overall cloud mean pairwise distance (expensive).

    :param x: input point cloud.
    :return: mean of pairwise distances.
    """
    diffs = x[:, None, :] - x[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    iu = np.triu_indices(x.shape[0], k=1)
    return float(dists[iu].mean())


def projection_residual(x, proj_fn):
    """
    Computes mean squared projection residual.

    :param x: input point cloud.
    :param proj_fn: projection function applied to ``x``.
    :return: mean squared residual between ``x`` and projected points.
    """
    xp = proj_fn(x)
    resid = x - xp
    return float(np.mean(np.sum(resid ** 2, axis=1)))


def total_persistence_h1(diagram_h1):
    """
    Computes total H1 persistence.

    :param diagram_h1: H1 persistence diagram with birth-death pairs.
    :return: sum of finite H1 lifetimes.
    """
    dgm = np.asarray(diagram_h1, dtype=float)
    if dgm.size == 0:
        return 0.0

    lifetimes = dgm[:, 1] - dgm[:, 0]
    lifetimes = lifetimes[np.isfinite(lifetimes)]
    if len(lifetimes) == 0:
        return 0.0
    return float(np.sum(lifetimes))


def max_persistence_h1(diagram_h1):
    """
    Computes maximum H1 persistence.

    :param diagram_h1: H1 persistence diagram with birth-death pairs.
    :return: largest finite H1 lifetime.
    """
    dgm = np.asarray(diagram_h1, dtype=float)
    if dgm.size == 0:
        return 0.0
    lifetimes = dgm[:, 1] - dgm[:, 0]
    lifetimes = lifetimes[np.isfinite(lifetimes)]
    if len(lifetimes) == 0:
        return 0.0
    return float(np.max(lifetimes))


def top5_persistence_h1(diagram_h1):
    """
    Computes sum of top five H1 persistences.

    :param diagram_h1: H1 persistence diagram with birth-death pairs.
    :return: sum of the five largest finite H1 lifetimes.
    """
    dgm = np.asarray(diagram_h1, dtype=float)
    if dgm.size == 0:
        return 0.0
    lifetimes = dgm[:, 1] - dgm[:, 0]
    lifetimes = lifetimes[np.isfinite(lifetimes)]
    if len(lifetimes) == 0:
        return 0.0
    lifetimes = np.sort(lifetimes)[::-1]
    return float(np.sum(lifetimes[:5]))

def mean_xy_radius(x):
    """
    Computes mean radius in the first two coordinates.

    :param x: input point cloud.
    :return: mean Euclidean norm of the first two coordinates.
    """
    return float(np.mean(np.linalg.norm(x[:, :2], axis=1)))


def betti_curve_from_diagram(_diagram, _grid):
    """
    Computes Betti curve from a persistence diagram.

    :param _diagram: persistence diagram with birth-death pairs.
    :param _grid: filtration values where the curve is evaluated.
    :return: Betti counts over the filtration grid.
    """
    dgm = np.asarray(_diagram, dtype=float)
    grid = np.asarray(_grid, dtype=float)

    if dgm.size == 0:
        return np.zeros_like(grid, dtype=float)

    births = dgm[:, 0][:, None]
    deaths = dgm[:, 1][:, None]
    alive = (births <= grid[None, :]) & (grid[None, :] < deaths)
    return alive.sum(axis=0).astype(float)


def betti_curve_area(_diagram, _grid):
    """
    Computes area under a Betti curve.

    :param _diagram: persistence diagram with birth-death pairs.
    :param _grid: filtration values where the curve is evaluated.
    :return: numerical integral of the Betti curve over the grid.
    """
    curve = betti_curve_from_diagram(_diagram, _grid)
    return float(np.trapz(curve, x=_grid))


def betti_curve_peak(_diagram, _grid):
    """
    Computes peak value of a Betti curve.

    :param _diagram: persistence diagram with birth-death pairs.
    :param _grid: filtration values where the curve is evaluated.
    :return: maximum Betti count over the grid.
    """
    curve = betti_curve_from_diagram(_diagram, _grid)
    return float(curve.max()) if len(curve) > 0 else 0.0


def betti_curve_change(_diagram, _ref_curve, _grid):
    """
    Computes change from a reference Betti curve.

    :param _diagram: persistence diagram with birth-death pairs.
    :param _ref_curve: reference Betti curve evaluated on ``_grid``.
    :param _grid: filtration values where the curves are evaluated.
    :return: area between the Betti curve and reference curve.
    """
    curve = betti_curve_from_diagram(_diagram, _grid)
    return float(np.trapz(np.abs(curve - _ref_curve), x=_grid))
