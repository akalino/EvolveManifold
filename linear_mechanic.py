""" Linear collapse: shrink orthogonal to a k-subspace. """
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from schedulers import get_schedule_value


Array = np.ndarray


@dataclass
class LinearSpectralParams:
    """ Dataclass to hold linear parameters."""
    k: int
    alpha_0: float = 1.0
    alpha_t: float = 0.02
    noise: float = 0.0
    schedule: str = "linear"


def step_linear_spectral(p: LinearSpectralParams, total_steps: int):
    """

    :param p:
    :param total_steps:
    :return:
    """

    def _step(x: Array, current_t: int, rng: np.random.Generator) -> Array:
        _, d = x.shape
        k = int(p.k)
        if not 0 <= k <= d:
            raise ValueError("k must satisfy 0 <= k <= d")

        alpha = get_schedule_value(
            p.schedule,
            total_steps,
            p.alpha_0,
            p.alpha_t,
            current_t + 1,
        )
        y = x.copy()
        if k < d:
            y[:, k:] = alpha * x[:, k:]
        if p.noise > 0:
            y = y + rng.normal(0.0, p.noise, size=y.shape)
        return y
    return _step
