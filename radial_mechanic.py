""" Radial collapse mechanisms."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np

from schedulers import get_schedule_value


Array = np.ndarray

@dataclass
class RadialParams:
    """Dataclass to hold radial collapse parameters."""
    schedule: str
    start_strength: float
    end_strength: float
    finish: int
    mover_frac: float = 1.0
    center_mode: str = "centroid"  # supports centroid, origin, fixed
    center: Optional[Array] = None
    target_radius: float = 0.0
    mode: str = "to_radius"
    fixed_indices: Optional[Array] = None
    seed: int = 0


Array = np.ndarray


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


def _compute_center(_x, _mode, _center):
    """
    Computes the center of the point cloud.

    _x: Input point cloud.
    _mode: Radial mechanism mode.
    _center: Center of the point cloud.
    :return: Center point.
    """
    if _mode == "centroid":
        return _x.mean(axis=0)
    if _mode == "origin":
        return np.zeros(_x.shape[1], dtype=_x.dtype)
    if _mode == "fixed":
        if _center is None:
            raise ValueError("center must be provided when center_mode='fixed'")
        return np.asarray(_center, dtype=_x.dtype)
    raise ValueError(f"Unknown center_mode: {_mode}")


def _safe_normalize(_v, _eps=1e-12):
    """
    Safely normalize a vector.
    :param _v: Input vector.
    :param _eps: Avoid div by zero.
    :return: Normalized vector.
    """
    norms = np.linalg.norm(_v, axis=1, keepdims=True)
    return _v / np.maximum(norms, _eps)


def step_radial(p: RadialParams):
    """
    Step radial collapse mechanism.
    :param p: Radial parameters dataclass.
    :return: _step, function for stepping.
    """

    def _step(x, current_t, rng) -> Array:
        x = np.asarray(x)
        xn = x.copy()
        lam = get_schedule_value(
            p.schedule,
            p.start_strength,
            p.end_strength,
            p.finish,
            current_t
        )

        center = _compute_center(x, p.center_mode, p.center)
        if p.fixed_indices is None:
            move_idx = _choose_indices(len(x), p.mover_frac, rng)
        else:
            move_idx = p.fixed_indices

        xc = x[move_idx] - center
        radii = np.linalg.norm(xc, axis=1, keepdims=True)
        if p.mode == "contract_to_center":
            # Simple centroid contraction
            xn[move_idx] = center + (1.0 - lam) * xc

        elif p.mode == "to_radius":
            # Preserve directions, move radii toward target_radius
            dirs = _safe_normalize(xc)
            new_radii = (1.0 - lam) * radii + lam * p.target_radius
            xn[move_idx] = center + dirs * new_radii

        else:
            raise ValueError(f"Unknown radial collapse mode: {p.mode}")

        return xn

    return _step


def radial_params_from_severity(
    severity: str,
    schedule: str,
    finish: int,
    mover_frac: float = 1.0,
    center_mode: str = "centroid",
    target_radius: float = 0.0,
    mode: str = "to_radius",
    seed: int = 0,
) -> RadialParams:
    if severity == "weak":
        start_strength, end_strength = 0.0, 0.25
    elif severity == "moderate":
        start_strength, end_strength = 0.0, 0.60
    elif severity == "strong":
        start_strength, end_strength = 0.0, 0.90
    else:
        raise ValueError(f"Unknown severity: {severity}")

    return RadialParams(
        schedule=schedule,
        start_strength=start_strength,
        end_strength=end_strength,
        finish=finish,
        mover_frac=mover_frac,
        center_mode=center_mode,
        target_radius=target_radius,
        mode=mode,
        seed=seed
    )
