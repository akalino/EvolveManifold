""" Contamination collapse mechanism. """
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

Array = np.ndarray

@dataclass
class ContaminationParams:
    """
    Dataclass for contamination parameters.
    """
    outlier_frac: float = 0.05
    outlier_mode: str = 'fixed'
    sigma: float = 2.0


def make_outlier_mask(_n, _outlier_frac, _rng):
    """

    :param _n: Total number of points.
    :param _outlier_frac: Pct of tagged outliers.
    :param _rng: random number generator.
    :return:
    """
    m = int(np.floor(np.clip(_outlier_frac, 0.0, 1.0) * _n))
    mask = np.zeros(_n, dtype=bool)
    if m > 0:
        idx = _rng.choice(_n, m, replace=False)
        mask[idx] = True
    return mask

def step_with_contamination(_base_fn, _params, _mask_seed=17):
    """

    :param _base_fn:
    :param _params:
    :param _mask_seed:
    :return:
    """
    mask_dict = {}

    def _step(_x, _t, _rng):
        n, d = _x.shape
        if "mask" not in mask_dict:
            _rng2 = np.random.default_rng(_mask_seed)
            mask_dict["mask"] = make_outlier_mask(n, _params.outlier_frac, _rng2)
        out_mask = mask_dict["mask"]
        in_mask = ~out_mask

        y = _x.copy()
        y_in = _base_fn(_x[in_mask], _t, _rng)
        y[in_mask] = y_in

        if _params.outlier_mode == 'fixed':
            y[out_mask] = _x[out_mask]
        elif _params.outlier_mode == 'messy':
            y[out_mask] = _x[out_mask] + _rng.normal(0.0, _params.sigma, size=(out_mask.sum(), d))
        else:
            raise ValueError("outlier_mode must be 'fixed' or 'messy'")
        return y

    return _step
