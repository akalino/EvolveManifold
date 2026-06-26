"""
IsoScore, anisotropy, and intrinsic-dimension metrics.

This module is intended for post-hoc evaluation over checkpointed point-cloud
trajectories. Inputs are point clouds with shape ``(n_points, ambient_dim)``.
"""

import numpy as np
from sklearn.decomposition import PCA

try:
    import skdim
    _HAS_SKDIM = True
except Exception:
    skdim = None
    _HAS_SKDIM = False


def _safe_float(x):
    """
    Convert a numeric value to a finite float.

    :param x: Numeric value.
    :return: Finite float or ``np.nan``.
    """
    try:
        val = float(x)
    except Exception:
        return np.nan
    if not np.isfinite(val):
        return np.nan
    return val


def _cov_eigvals(x):
    """
    Compute covariance eigenvalues in descending order.

    :param x: Point cloud of shape ``(n_points, ambient_dim)``.
    :return: Eigenvalues sorted descending.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        return np.array([], dtype=float)

    xc = x - x.mean(axis=0, keepdims=True)
    cov = (xc.T @ xc) / max(x.shape[0] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    return np.sort(eigvals)[::-1]


def iso_score(x, eps=1e-12):
    """
    Compute IsoScore-style isotropy score from PCA/covariance energy.

    Higher values indicate more isotropic variance; lower values indicate
    stronger anisotropy or collapse toward fewer directions.

    :param x: Point cloud of shape ``(n_points, ambient_dim)``.
    :param eps: Numerical floor.
    :return: IsoScore scalar.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or min(x.shape) < 2:
        return np.nan

    n_comp = min(x.shape[0], x.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(x)

    cov_diag = np.asarray(pca.explained_variance_, dtype=float)
    norm = np.linalg.norm(cov_diag)
    if norm <= eps:
        return np.nan

    n = len(cov_diag)
    cov_diag_normalized = (cov_diag * np.sqrt(n)) / norm
    iso_diag = np.ones(n)

    l2_norm = np.linalg.norm(cov_diag_normalized - iso_diag)
    denom = np.sqrt(2 * (n - np.sqrt(n)))
    if denom <= eps:
        return np.nan

    iso_defect = l2_norm / denom
    score = ((n - (iso_defect ** 2) * (n - np.sqrt(n))) ** 2 - n) / (n * (n - 1))
    return _safe_float(score)


def anisotropy_ratio(x, eps=1e-12):
    """
    Compute largest-eigenvalue-to-mean-eigenvalue ratio.

    :param x: Point cloud of shape ``(n_points, ambient_dim)``.
    :param eps: Numerical floor.
    :return: Anisotropy ratio.
    """
    eigvals = _cov_eigvals(x)
    if eigvals.size == 0:
        return np.nan

    total = float(eigvals.sum())
    if total <= eps:
        return np.nan

    mean_eigval = total / len(eigvals)
    return _safe_float(eigvals[0] / max(mean_eigval, eps))


def id_two_nn(x):
    """
    Estimate intrinsic dimension using TwoNN.

    :param x: Point cloud of shape ``(n_points, ambient_dim)``.
    :return: Intrinsic-dimension estimate.
    """
    if not _HAS_SKDIM:
        return np.nan

    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] < 5:
        return np.nan

    try:
        est = skdim.id.TwoNN().fit(x)
        return _safe_float(est.dimension_)
    except Exception:
        return np.nan


def id_mle(x, mle_k=20):
    """
    Estimate intrinsic dimension using skdim MLE.

    :param x: Point cloud of shape ``(n_points, ambient_dim)``.
    :param mle_k: Neighborhood size for the MLE estimator.
    :return: Intrinsic-dimension estimate.
    """
    if not _HAS_SKDIM:
        return np.nan

    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] < 5:
        return np.nan

    try:
        k_eff = max(2, min(int(mle_k), x.shape[0] - 1))
        est = skdim.id.MLE(K=k_eff).fit(x)
        return _safe_float(est.dimension_)
    except Exception:
        return np.nan


def compute_iso_id_metrics(x, mle_k=20):
    """
    Compute all IsoScore and intrinsic-dimension metrics.

    :param x: Point cloud of shape ``(n_points, ambient_dim)``.
    :param mle_k: Neighborhood size for MLE intrinsic dimension.
    :return: Flat dictionary of scalar metrics.
    """
    return {
        "iso_score": iso_score(x),
        "anisotropy_ratio": anisotropy_ratio(x),
        "id_two_nn": id_two_nn(x),
        "id_mle": id_mle(x, mle_k=mle_k),
        "id_mle_k": int(max(2, min(int(mle_k), np.asarray(x).shape[0] - 1))),
        "id_skdim_available": int(_HAS_SKDIM),
    }
