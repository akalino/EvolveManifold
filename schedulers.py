""" Linear and exponential per-epoch scheduling."""
import math

def exp_decay(_end, _eps_0, _eps_t, _current_t):
    """

    :param _end: Ending epoch (T).
    :param _eps_0: Initial epsilon.
    :param _eps_t: Final epsilon.
    :param _current_t: The current epoch (t)
    :return:
    """
    if _end <= 0:
        return _eps_t
    r = (_eps_t / _eps_0) ** (1 / _end)
    return _eps_0 * (r ** _current_t)


def linear_decay(_end, _a_0, _a_t, _current_t):
    """

    :param _end: Ending epoch (T).
    :param _a_0: Initial slope.
    :param _a_t: Slope at time (t).
    :param _current_t: Current step (t).
    :return:
    """
    if _end <= 0:
        return _a_t
    return _a_0 + (_a_t - _a_0) * (_current_t / _end)

def sigmoid_decay(
    _end,
    _a_0,
    _a_t,
    _current_t,
    _sharpness: float = 10.0,
):
    """
    :param _end: Ending epoch (T).
    :param _a_0: Initial slope.
    :param _a_t: Slope at time (t).
    :param _current_t: Current step (t).
    :param _sharpness: Sharpening factor.
    """
    if _end <= 0:
        return _a_t

    s = _current_t / _end
    z = 1.0 / (1.0 + math.exp(-_sharpness * (s - 0.5)))
    z0 = 1.0 / (1.0 + math.exp(_sharpness * 0.5))
    z1 = 1.0 / (1.0 + math.exp(-_sharpness * 0.5))
    w = (z - z0) / (z1 - z0)

    return _a_0 + (_a_t - _a_0) * w


def get_schedule_value(_schedule, _end, _start, _finish, _current_t):
    """
    """
    name = str(_schedule).lower()

    if name == "linear":
        return linear_decay(_end, _start, _finish, _current_t)
    if name in {"exp", "exponential"}:
        return exp_decay(_end, _start, _finish, _current_t)
    if name in {"sigmoid", "delayed", "delayed_exp", "sigmoid/delayed"}:
        return sigmoid_decay(_end, _start, _finish, _current_t)

    raise ValueError(f"Unknown schedule type: {_schedule}")