from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DetectionConfig:
    threshold: float = 0.10
    window: int = 2
    eps: float = 1e-12
    return_T_plus_1: bool = True


def collapse_score(
    values: Iterable[float],
    *,
    increases_with_collapse: bool,
    eps: float = 1e-12,
) -> np.ndarray:
    x = np.asarray(list(values), dtype=float)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D metric series, got shape {x.shape}")

    baseline = x[0]
    denom = abs(baseline) + eps

    if increases_with_collapse:
        return (x - baseline) / denom
    return (baseline - x) / denom


def detection_time_from_score(
    score: Iterable[float],
    *,
    threshold: float = 0.10,
    window: int = 2,
    return_T_plus_1: bool = True,
) -> float:
    s = np.asarray(list(score), dtype=float)
    if s.ndim != 1:
        raise ValueError(f"Expected 1D score series, got shape {s.shape}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    T = len(s)
    last_start = T - window + 1
    for t in range(max(0, last_start)):
        if np.all(s[t : t + window] >= threshold):
            return float(t)

    return float(T) if return_T_plus_1 else math.inf


def detection_time(
    values: Iterable[float],
    *,
    increases_with_collapse: bool,
    config: Optional[DetectionConfig] = None,
) -> float:
    cfg = config or DetectionConfig()
    score = collapse_score(
        values,
        increases_with_collapse=increases_with_collapse,
        eps=cfg.eps,
    )
    return detection_time_from_score(
        score,
        threshold=cfg.threshold,
        window=cfg.window,
        return_T_plus_1=cfg.return_T_plus_1,
    )


def median_detection_time_across_seeds(
    df: pd.DataFrame,
    *,
    metric_col: str,
    time_col: str = "epoch",
    seed_col: str = "seed",
    increases_with_collapse: bool,
    group_cols: Optional[list[str]] = None,
    config: Optional[DetectionConfig] = None,
) -> pd.DataFrame:
    cfg = config or DetectionConfig()
    group_cols = group_cols or []

    required = set(group_cols + [seed_col, time_col, metric_col])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    seed_groups = group_cols + [seed_col]
    rows = []

    for keys, subdf in df.groupby(seed_groups, dropna=False):
        subdf = subdf.sort_values(time_col)
        t_detect = detection_time(
            subdf[metric_col].to_numpy(),
            increases_with_collapse=increases_with_collapse,
            config=cfg,
        )

        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(seed_groups, keys))
        row["t_detect"] = t_detect
        rows.append(row)

    det_df = pd.DataFrame(rows)

    return (
        det_df.groupby(group_cols, dropna=False)["t_detect"]
        .agg(
            median_t_detect="median",
            mean_t_detect="mean",
            std_t_detect="std",
            n_seeds="count",
        )
        .reset_index()
    )