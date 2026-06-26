"""
Landmark selection utilities for OT tracing.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def random_landmarks(n, n_landmarks, seed):
    """
    Select random landmarks at each checkpoint.

    :param n: total point cloud points.
    :param n_landmarks: selected number of landmarks.
    :param seed: random seed.
    :return: indexed random points.
    """
    rng = np.random.default_rng(seed)
    m = min(int(n_landmarks), int(n))
    return np.sort(rng.choice(n, size=m, replace=False))


def fps_landmarks(x, n_landmarks, seed):
    """
    FPS landmark selection.

    :param x: point cloud.
    :param n_landmarks: number of landmarks.
    :param seed: random seed.
    :return: indexed FPS points.
    """
    n = x.shape[0]
    m = min(int(n_landmarks), int(n))

    rng = np.random.default_rng(seed)
    idx = np.empty(m, dtype=int)
    idx[0] = int(rng.integers(0, n))

    d2 = np.sum((x - x[idx[0]]) ** 2, axis=1)

    for k in range(1, m):
        idx[k] = int(np.argmax(d2))
        new_d2 = np.sum((x - x[idx[k]]) ** 2, axis=1)
        d2 = np.minimum(d2, new_d2)

    return np.sort(idx)


def stratified_label_landmarks(df, n_landmarks, seed, label_col):
    """
    Basic proportional stratified sampling by label.
    Assumes row order matches the checkpoint coordinate matrix.

    :param df: dataframe of checkpoints.
    :param n_landmarks: number of landmarks.
    :param seed: random seed.
    :param label_col: column name for labels.
    :return: selected label indices for landmarking.
    """

    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    rng = np.random.default_rng(seed)
    n = len(df)
    m = min(int(n_landmarks), int(n))

    counts = df[label_col].value_counts(dropna=False)
    selected = []

    for label, count in counts.items():
        frac = count / float(n)
        take = max(1, int(round(frac * m)))
        label_indices = np.flatnonzero(df[label_col].to_numpy() == label)
        take = min(take, len(label_indices))
        selected.extend(rng.choice(label_indices, size=take, replace=False).tolist())

    selected = np.array(sorted(set(selected)), dtype=int)

    if len(selected) > m:
        selected = np.sort(rng.choice(selected, size=m, replace=False))

    if len(selected) < m:
        missing = m - len(selected)
        pool = np.setdiff1d(np.arange(n), selected)
        extra = rng.choice(pool, size=missing, replace=False)
        selected = np.sort(np.concatenate([selected, extra]))

    return selected


def landmark_coverage_stats(x, landmark_idx):
    """
    Coverage statistics for landmark strategies.

    :param x: point cloud.
    :param landmark_idx: indices of selected landmarks.
    :return: dict of statistics.
    """
    landmarks = x[landmark_idx]
    d = cdist(x, landmarks, metric="euclidean")
    nearest = np.min(d, axis=1)

    q = np.quantile(nearest, [0.5, 0.9, 0.95, 0.99, 1.0])

    return {
        "landmark_coverage_mean": float(np.mean(nearest)),
        "landmark_coverage_std": float(np.std(nearest)),
        "landmark_coverage_median": float(q[0]),
        "landmark_coverage_q90": float(q[1]),
        "landmark_coverage_q95": float(q[2]),
        "landmark_coverage_q99": float(q[3]),
        "landmark_coverage_max": float(q[4]),
    }


def choose_landmarks(x, n_landmarks, seed, method, df, label_col):
    """
    landmark selection function.

    :param x: point cloud.
    :param n_landmarks: number of landmarks.
    :param seed: random seed.
    :param method: landmark selection method.
    :param df: trajectory data frame.
    :param label_col: column containing labels.
    :return: landmark indices based on method.
    """
    if method == "random":
        return random_landmarks(x.shape[0], n_landmarks, seed)

    if method == "fps":
        return fps_landmarks(x, n_landmarks, seed)

    if method == "stratified_label":
        if df is None:
            raise ValueError("stratified_label requires the source checkpoint dataframe.")
        return stratified_label_landmarks(df, n_landmarks, seed, label_col)

    raise ValueError(f"Unknown landmark method: {method}")
