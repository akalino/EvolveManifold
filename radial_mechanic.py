"""Radial collapse mechanisms."""

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

    # Absolute target radius. If None, use target_radius_frac times each
    # point's initial radius from the center.
    target_radius: Optional[float] = None

    # Nonzero final radius fraction used by default. This prevents strong
    # radial collapse from degenerating to a single point.
    target_radius_frac: float = 0.15

    # supports contract_to_center, to_radius
    mode: str = "to_radius"
    fixed_indices: Optional[Array] = None
    seed: int = 0


def _choose_indices(_n, _mover_frac, _rng):
    """Chooses the indices of points moved."""
    m = max(1, int(round(_mover_frac * _n)))
    return _rng.choice(_n, size=m, replace=False)


def _compute_center(_x, _mode, _center):
    """Computes the center of the point cloud."""
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
    """Safely normalize vectors row-wise."""
    norms = np.linalg.norm(_v, axis=1, keepdims=True)
    return _v / np.maximum(norms, _eps)


def step_radial(p: RadialParams):
    """
    Build a radial collapse step function.

    Important design choice:
    - The center and initial radii are anchored from the first point cloud seen
      by this closure.
    - Later checkpoints interpolate toward a nonzero final radius, rather than
      repeatedly contracting the already-contracted cloud to zero.
    """

    state = {
        "x0": None,
        "center0": None,
        "radii0": None,
        "dirs0": None,
        "fixed_indices": None,
    }

    def _initialize(x, rng):
        x0 = np.asarray(x).copy()
        center0 = _compute_center(x0, p.center_mode, p.center)

        if p.fixed_indices is None:
            fixed_indices = _choose_indices(len(x0), p.mover_frac, rng)
        else:
            fixed_indices = np.asarray(p.fixed_indices, dtype=int)

        xc0 = x0[fixed_indices] - center0
        radii0 = np.linalg.norm(xc0, axis=1, keepdims=True)
        dirs0 = _safe_normalize(xc0)

        state["x0"] = x0
        state["center0"] = center0
        state["radii0"] = radii0
        state["dirs0"] = dirs0
        state["fixed_indices"] = fixed_indices

    def _step(x, current_t, rng) -> Array:
        x = np.asarray(x)

        if state["x0"] is None:
            _initialize(x, rng)

        x0 = state["x0"]
        center0 = state["center0"]
        radii0 = state["radii0"]
        dirs0 = state["dirs0"]
        move_idx = state["fixed_indices"]

        lam = get_schedule_value(
            p.schedule,
            p.finish,
            p.start_strength,
            p.end_strength,
            current_t + 1,
        )
        lam = float(np.clip(lam, 0.0, 1.0))

        xn = x0.copy()

        if p.mode == "contract_to_center":
            # Contract toward a nonzero radius fraction instead of the exact
            # center. This avoids degenerating the whole cloud to one point.
            final_radii = p.target_radius_frac * radii0
            new_radii = (1.0 - lam) * radii0 + lam * final_radii
            xn[move_idx] = center0 + dirs0 * new_radii

        elif p.mode == "to_radius":
            # Preserve initial directions and move initial radii toward either:
            # - an absolute target radius, or
            # - a fraction of each point's initial radius.
            if p.target_radius is None:
                target_radii = p.target_radius_frac * radii0
            else:
                target_radii = np.full_like(radii0, float(p.target_radius))

            new_radii = (1.0 - lam) * radii0 + lam * target_radii
            xn[move_idx] = center0 + dirs0 * new_radii

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
    target_radius: Optional[float] = None,
    mode: str = "to_radius",
    seed: int = 0,
) -> RadialParams:
    if severity == "weak":
        start_strength, end_strength = 0.0, 1.0
        target_radius_frac = 0.70
    elif severity in {"moderate", "medium"}:
        start_strength, end_strength = 0.0, 1.0
        target_radius_frac = 0.40
    elif severity == "strong":
        start_strength, end_strength = 0.0, 1.0
        target_radius_frac = 0.15
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
        target_radius_frac=target_radius_frac,
        mode=mode,
        seed=seed,
    )
