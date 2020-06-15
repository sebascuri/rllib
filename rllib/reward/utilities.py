"""Tools to help define differentiable reward functions."""

import torch


def gaussian(x, value_at_1):
    """Apply an un-normalized Gaussian function with zero mean and scaled variance.

    Parameters
    ----------
    x : The points at which to evaluate_agent the Gaussian
    value_at_1: The reward magnitude when x=1. Needs to be 0 < value_at_1 < 1.
    """
    if type(value_at_1) is not torch.Tensor:
        value_at_1 = torch.tensor(value_at_1)
    scale = torch.sqrt(-2 * torch.log(value_at_1))
    return torch.exp(-0.5 * (x * scale) ** 2)


def tolerance(x, lower, upper, margin=None):
    """Apply a tolerance function with optional smoothing.

    Can be used to design (smoothed) box-constrained reward functions.

    A tolerance function is returns 1 if x is in [lower, upper].
    If it is outside, it decays exponentially according to a margin.

    Parameters
    ----------
    x : the value at which to evaluate_agent the sparse reward.
    lower: The lower bound of the tolerance function.
    upper: The upper bound of the tolerance function.
    margin: A margin over which to smooth out the box-reward.
        If a positive margin is provided, uses a `gaussian` smoothing on the boundary.
    """
    if margin is None or margin == 0.0:
        in_bounds = (lower <= x) & (x <= upper)
        return in_bounds.type(torch.get_default_dtype())
    else:
        assert margin > 0
        diff = 0.5 * (upper - lower)
        mid = lower + diff

        # Distance is positive only outside the bounds
        distance = torch.abs(x - mid) - diff
        return gaussian(torch.relu(distance * (1 / margin)), value_at_1=0.1)
