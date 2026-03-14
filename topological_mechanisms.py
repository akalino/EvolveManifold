"""Mechanisms to force topological collapse"""
import numpy as np

from schedulers import get_schedule_value


@dataclass
class HoleFillParams:
    r0: float = 0.0
    rt: float = 0.8
    schedule: str = "linear"
    noise: float = 0.0


@dataclass
class PinchParams:
    strength_0: float = 0.0
    strength_t: float = 1.0
    theta0: float = 0.0
    sigma: float = 0.35
    schedule: str = "linear"
    noise: float = 0.0


@dataclass
class BridgeParams:
    strength_0: float = 0.0
    strength_t: float = 1.0
    theta0: float = 0.0
    sigma: float = 0.35
    bridge_half_width: float = 0.15
    schedule: str = "linear"
    noise: float = 0.0

# helpers

def _angle_diff(theta, theta0):
    d = theta - theta0
    return np.arctan2(np.sin(d), np.cos(d))


def _sector_weight(theta, theta0, sigma):
    d = _angle_diff(theta, theta0)
    return np.exp(-(d ** 2) / (2.0 * sigma ** 2))


def _xy_radius_theta(x):
    xy = x[:, :2]
    r = np.linalg.norm(xy, axis=1)
    theta = np.arctan2(xy[:, 1], xy[:, 0])
    return xy, r, theta

# step mechanics

def step_hole_fill(p: HoleFillParams, total_steps: int):
    """
    Pull points inward to the first two coordinates. Works well with ring/torus.
    """
    def _step(x, current_t, rng):
        y = x.copy()
        lam = get_schedule_value(p.schedule,
                                 total_steps,
                                 p.r0,
                                 p.rt,
                                 current_t + 1)
        xy, r, _ = _xy_radius_theta(y)
        target_r = np.maximum(r - lam, 0.0)
        scale = np.ones_like(r)
        nz = r > 1e-12
        scale[nz] = target_r[nz] / r[nz]
        y[:, :2] = xy * scale[:, None]

        if p.noise > 0:
            y = y + rng.normal(0.0, p.noise, size=y.shape)

        return y
    return _step


def step_loop_pinch(p: PinchParams, total_steps: int):
    """
    Pinch a localized angular sector toward the origin in the first two
    coordinates to kill loop structure locally.
    """
    def _step(x, current_t, rng):
        y = x.copy()

        lam = get_schedule_value(
            p.schedule,
            total_steps,
            p.strength_0,
            p.strength_t,
            current_t + 1,
        )

        xy, _, theta = _xy_radius_theta(y)
        w = _sector_weight(theta, p.theta0, p.sigma)

        shrink = 1.0 - lam * w
        y[:, :2] = xy * shrink[:, None]

        if p.noise > 0:
            y = y + rng.normal(0.0, p.noise, size=y.shape)

        return y

    return _step


def step_bridge_across_hole(p: BridgeParams, total_steps: int):
    """
    Pull two opposite sectors toward the line joining them, creating a bridge
    across the hole in the first two coordinates.
    """
    def _step(x, current_t, rng):
        y = x.copy()

        lam = get_schedule_value(
            p.schedule,
            total_steps,
            p.strength_0,
            p.strength_t,
            current_t + 1,
        )

        xy, _, theta = _xy_radius_theta(y)

        w1 = _sector_weight(theta, p.theta0, p.sigma)
        w2 = _sector_weight(theta, p.theta0 + np.pi, p.sigma)
        w = np.maximum(w1, w2)

        v = np.array([np.cos(p.theta0), np.sin(p.theta0)])
        proj = (xy @ v)[:, None] * v[None, :]
        ortho = xy - proj

        pull = np.clip(lam * w, 0.0, 1.0)
        y[:, :2] = proj + (1.0 - pull)[:, None] * ortho

        if p.noise > 0:
            y = y + rng.normal(0.0, p.noise, size=y.shape)

        return y

    return _step
