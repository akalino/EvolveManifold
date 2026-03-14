"""Checkpoint manager for saving point-cloud states during evolution."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass
class CheckpointManager:
    """
    Manages snapshots of point cloud states during evolution.
    """
    _root_dir: str = "evolve_checkpoints"
    _experiment: str = "default"
    _mechanism: str = "unknown"
    _model: str = "unknown"
    _every: int = 10
    _save_rng: bool = False
    _overwrite: bool = True

    def __post_init__(self):
        if self._every <= 0:
            raise ValueError("Save step must be positive")
        self._run_dir = os.path.join(
            self._root_dir,
            self._experiment,
            self._mechanism,
            self._model
        )

        if os.path.exists(self._run_dir):
            if self._overwrite:
                for root, dirs, files in os.walk(self._run_dir, topdown=False):
                    for f in files:
                        os.remove(os.path.join(root, f))
                    for d in dirs:
                        os.rmdir(os.path.join(root, d))
            else:
                raise FileExistsError(
                    f"Run directory already exists: {self._run_dir}. "
                    "Set _overwrite=True or provide a unique _run_id."
                )
        os.makedirs(self._run_dir, exist_ok=True)

    @property
    def run_dir(self):
        """
        Return run directory.
        """
        return self._run_dir

    def _ckpt_path(self, _epoch):
        """
        :param _epoch:
        """
        return os.path.join(self._run_dir, f"ckpt_epoch_{_epoch:04d}.pkl")

    def save(self, _x, _epoch):
        """
        :param _x: point cloud state
        :param _epoch:
        """
        payload = {"x": np.asarray(_x, dtype=float),
                   "epoch": int(_epoch),
                   "experiment": self._experiment,
                   "mechanism": self._mechanism,
                   "model": self._model}
        with open(self._ckpt_path(int(_epoch)), "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def soft_save(self, _x, _epoch, _force=False):
        """
        Check save step pattern.

        :param _x: Point cloud.
        :param _epoch: Checkpoint number.
        :param _force: Force save step.
        """
        if _force or (_epoch == 0) or (int(_epoch) % self._every ==  0):
            self.save(_x, _epoch)

    def load(self, _epoch):
        """
        :param _epoch:
        """
        with open(self._ckpt_path(int(_epoch)), "rb") as f:
            return pickle.load(f)
