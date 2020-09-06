"""Utilities for the transformers."""
import torch


def rescale(tensor, scale):
    """Rescale an array by multiplying it by scale."""
    if tensor.dim() < 2 or scale.shape[-1] != tensor.shape[-2]:
        return tensor
    else:
        return scale @ tensor


def update_mean(old_mean, old_count, new_mean, new_count):
    """Update mean based on a new batch of data.

    Parameters
    ----------
    old_mean : array_like
    old_count : int
    new_mean : array_like
    new_count : int

    References
    ----------
    Uses a modified version of Welford's algorithm, see
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    """
    total = old_count + new_count
    mean = (old_count * old_mean + new_count * new_mean) / total
    return mean


def update_var(old_mean, old_var, old_count, new_mean, new_var, new_count):
    """Update mean and variance statistics based on a new batch of data.

    Parameters
    ----------
    old_mean : array_like
    old_var : array_like
    old_count : int
    new_mean : array_like
    new_var : array_like
    new_count : int

    References
    ----------
    Uses a modified version of Welford's algorithm, see
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    """
    delta = new_mean - old_mean
    total = old_count + new_count
    if total <= 1:
        return torch.ones_like(new_var)

    if old_count <= 1:
        old_m = torch.zeros_like(old_var)
    else:
        old_m = old_var * (old_count - 1)

    if new_count <= 1:
        new_m = torch.zeros_like(new_var)
    else:
        new_m = new_var * (new_count - 1)

    m2 = old_m + new_m + delta ** 2 * (old_count * new_count) / total
    return m2 / (total - 1.0)
