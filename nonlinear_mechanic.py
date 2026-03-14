""" Nonlinear collapse mechanism. """
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from schedulers import get_schedule_value


@dataclass
class NonLinearParams:
    """Dataclass to hold nonlinear parameters."""
    eps_0: float = 0.5
    eps_t: float = 0.01
    relax: float = 1.0
    schedule: str = "exponential"


def step_nonlinear_projection(_proj_fn, _params: NonLinearParams, _t):
    def _step(_x, current_t, _rng):
        p = _proj_fn(_x)
        eps = get_schedule_value(
            _params.schedule,
            _t,
            _params.eps_0,
            _params.eps_t,
            current_t + 1,
        )

        y = (1.0 - _params.relax) * _x + _params.relax * p
        y = y + eps * _rng.normal(size=_x.shape)
        return y

    return _step