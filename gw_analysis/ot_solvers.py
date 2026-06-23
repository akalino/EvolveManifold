"""
Using multiple solvers from POT for comparison/fidelity.
"""

import numpy as np

from scipy.spatial.distance import cdist

import ot


def uniform_weights(n):
    """
    Create an array of uniform weights for n points.

    :param n: number of points.
    :return: ndarray of n uniform weights.
    """
    return np.ones(n, dtype=np.float64)


def entropic_gw(c1, c2, eps, max_iter, tol):
    """
    Entropic regularized GW.

    :param c1: source cost matrix.
    :param c2: target cost matrix.
    :param eps: entropic regularization param.
    :param max_iter: number of iterations.
    :param tol: floating point tolerance.
    :return: EGW value, float.
    """
    p = uniform_weights(c1.shape[0])
    q = uniform_weights(c2.shape[0])

    val = ot.gromov.entropic_gromov_wasserstein2(
        c1, c2, p, q, loss_fn="square_loss",
        epsilon=eps, max_iter=max_iter, tol=tol,
        verbose=False
    )
    return float(val)


def gw(c1, c2, max_iter, tol):
    """
    High fidelity but slower GW computation (no Sinkhorn).

    :param c1: source cost matrix.
    :param c2: target cost matrix.
    :param max_iter: number of iterations.
    :param tol: floating point tolerance.
    :return: GW value, float.
    """
    p = uniform_weights(c1.shape[0])
    q = uniform_weights(c2.shape[0])

    val = ot.gromov.gromov_wasserstein2(
        c1, c2, p, q, loss_fn="square_loss",
        max_iter=max_iter, tol_rel=tol, tol_abs=tol,
        verbose=False
    )
    return float(val)


def fused_gw(x1, x2, c1, c2, alpha, eps, max_iter, tol):
    """
    Fused GW combines feature-space cost M with internal relational costs C1, C2.
    Plain GW compares **relational geometry**. But for these synthetic trajectories,
    point identities often persist across checkpoints, and coordinate displacement
    also matters. Fused GW combines feature-space costs with relational costs.
    Alpha near 1 emphasizes relational GW, alpha near 0 emphasizes direct
    feature/coordinate GW.

    :param x1: PC1.
    :param x2: PC2.
    :param c1: source cost matrix.
    :param c2: target cost matrix.
    :param alpha:
    :param eps: entropic regularization param.
    :param max_iter: number of iterations.
    :param tol: floating point tolerance.
    :return: fused GW, float.
    """
    p = uniform_weights(c1.shape[0])
    q = uniform_weights(c2.shape[0])

    m = cdist(x1, x2, metric="euclidean").astype(np.float64, copy=False)

    if epsilon is not None and epsilon > 0:
        value = ot.gromov.entropic_fused_gromov_wasserstein2(
            m,
            c1,
            c2,
            p,
            q,
            loss_fun="square_loss",
            alpha=alpha,
            epsilon=epsilon,
            max_iter=max_iter,
            tol=tol,
            verbose=False,
        )
    else:
        value = ot.gromov.fused_gromov_wasserstein2(
            m,
            c1,
            c2,
            p,
            q,
            loss_fun="square_loss",
            alpha=alpha,
            max_iter=max_iter,
            tol_rel=tol,
            tol_abs=tol,
            verbose=False,
        )

    return float(value)


def solve_ot_distance(
        method,
        x1, x2, c1, c2, eps, alpha, max_iter, tol):
    """
    Common solver for all methods.

    :param method: OT compute method.
    :param x1: PC1.
    :param x2: PC2.
    :param c1: source cost matrix.
    :param c2: target cost matrix.
    :param eps: entropic regularization param.
    :param alpha:
    :param max_iter: number of iterations.
    :param tol: tolerance.
    :return: dict of trajectory worker diagnostics.
    """
    try:
        if method == "entropic_gw":
            value = entropic_gw(c1, c2, eps, max_iter, tol)
        elif method == "gw":
            value = gw(c1, c2, max_iter, tol)
        elif method == "fused_gw":
            value = fused_gw(x1, x2, c1, c2, alpha, eps, max_iter, tol)
        else:
            raise ValueError(f"Unknown OT method: {method}")

        return {
            "ot_value": float(value),
            "ot_distance": float(np.sqrt(max(value, 0.0))),
            "ot_status": "ok",
            "ot_error": None,
        }

    except Exception as exc:
        return {
            "ot_value": float("nan"),
            "ot_distance": float("nan"),
            "ot_status": "failed",
            "ot_error": repr(exc),
        }
