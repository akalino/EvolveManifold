""" Dataclass for running point move trajectories."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from tqdm import tqdm


Array = np.ndarray

@dataclass
class Trajectory:
    """
    Dataclass for running point move trajectories.
    """
    _xs: List[Array]
    _displace: List[Array]
    _movers: List[Array]


def dynamics(_x0, _total_step, _step_fn, _mover_frac=0.05, _seed=None, _chkpt=None):
    """
    Builds point cloud movement dynamics.

    :param _x0:
    :param _total_step:
    :param _step_fn:
    :param _mover_frac:
    :param _seed:
    :param _chkpt:
    :return:
    """
    if _chkpt is not None:
        _chkpt.soft_save(_x0, 0, True)

    r_num = np.random.default_rng(seed=_seed)
    x = np.asarray(_x0, dtype=float).copy()
    n, _ = x.shape

    xs = [x.copy()]
    disps = []
    movers = []

    p = float(np.clip(_mover_frac, 0.0, 1.0))

    if p <= 0.0:
        mover_mask = np.zeros(n, dtype=bool)
    elif p >= 1.0:
        mover_mask = np.ones(n, dtype=bool)
    else:
        k_move = max(1, int(np.floor(n * p)))
        idx = r_num.choice(n, size=k_move, replace=False)
        mover_mask = np.zeros(n, dtype=bool)
        mover_mask[idx] = True

    for t in tqdm(range(_total_step)):
        xn = _step_fn(x, t, r_num)
        delta = np.linalg.norm(xn - x, axis=1)

        x_next = x.copy()
        x_next[mover_mask] = xn[mover_mask]

        disps.append(delta)
        movers.append(mover_mask.copy())
        x = x_next
        xs.append(x.copy())

        if _chkpt is not None:
            _chkpt.soft_save(x, t + 1, False)

    return Trajectory(xs, disps, movers)
