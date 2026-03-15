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
