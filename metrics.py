import numpy as np


def effective_rank(x, eps=1e-12):
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
    xc = x - x.mean(axis=0, keepdims=True)
    cov = (xc.T @ xc) / xc.shape[0]
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0.0, None)[::-1]

    total = eigvals.sum()
    if total <= 0:
        return 0.0

    return float(eigvals[:k].sum() / total)


def mean_pairwise_distance(x):
    diffs = x[:, None, :] - x[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    iu = np.triu_indices(x.shape[0], k=1)
    return float(dists[iu].mean())


def projection_residual(x, proj_fn):
    xp = proj_fn(x)
    resid = x - xp
    return float(np.mean(np.sum(resid ** 2, axis=1)))


def total_persistence_h1(diagram_h1):
    dgm = np.asarray(diagram_h1, dtype=float)
    if dgm.size == 0:
        return 0.0

    lifetimes = dgm[:, 1] - dgm[:, 0]
    lifetimes = lifetimes[np.isfinite(lifetimes)]
    if len(lifetimes) == 0:
        return 0.0
    return float(np.sum(lifetimes))

def mean_xy_radius(x):
    return float(np.mean(np.linalg.norm(x[:, :2], axis=1)))


def betti_curve_from_diagram(_diagram, _grid):
    dgm = np.asarray(_diagram, dtype=float)
    grid = np.asarray(_grid, dtype=float)

    if dgm.size == 0:
        return np.zeros_like(grid, dtype=float)

    births = dgm[:, 0][:, None]
    deaths = dgm[:, 1][:, None]
    alive = (births <= grid[None, :]) & (grid[None, :] < deaths)
    return alive.sum(axis=0).astype(float)


def betti_curve_area(_diagram, _grid):
    curve = betti_curve_from_diagram(_diagram, _grid)
    return float(np.trapz(curve, x=_grid))


def betti_curve_peak(_diagram, _grid):
    curve = betti_curve_from_diagram(_diagram, _grid)
    return float(curve.max()) if len(curve) > 0 else 0.0


def betti_curve_change(_diagram, _ref_curve, _grid):
    curve = betti_curve_from_diagram(_diagram, _grid)
    return float(np.trapz(np.abs(curve - _ref_curve), x=_grid))

