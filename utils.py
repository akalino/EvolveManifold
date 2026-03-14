"""" Utilities for point cloud manipulation."""
import numpy as np

Array = np.ndarray


def random_orthogonal(d: int, rng: np.random.Generator) -> Array:
    """Random orthogonal matrix via QR."""
    a = rng.normal(size=(d, d))
    q, _ = np.linalg.qr(a)
    return q


def rotate_cloud(x: Array, q: Array) -> Array:
    """Apply rotation Q (dxd) to points X (nxd)."""
    return np.asarray(x) @ np.asarray(q)
