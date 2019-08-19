"""Utilities for the transformers."""
import numpy as np
import torch


def get_backend(array):
    """Get the backend of a given array."""
    if isinstance(array, torch.Tensor):
        return torch
    else:
        return np


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


def update_var(old_mean, old_var, old_count, new_mean, new_var, new_count, biased=True):
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

    if not biased:
        old_c = old_count - 1 if old_count > 0 else 0
        new_c = new_count - 1 if new_count > 0 else 0
    else:
        old_c = old_count
        new_c = new_count

    old_m = old_var * old_c
    new_m = new_var * new_c

    m2 = old_m + new_m + delta ** 2 * (old_count * new_count / total)

    if not biased:
        return m2 / (total - 1)
    else:
        return m2 / total


def normalize(array, mean, variance, preserve_origin=False):
    """Normalize an array.

    Parameters
    ----------
    array : array_like
    mean : array_like
    variance : array_like
    preserve_origin : bool, optional
        Whether to retain the origin (sign) of the data.
    """
    backend = get_backend(array)
    if preserve_origin:
        scale = backend.sqrt(variance + mean ** 2)
        return array / scale
    else:
        return (array - mean) / backend.sqrt(variance)


def denormalize(array, mean, variance, preserve_origin=False):
    """Denormalize an array.

    Parameters
    ----------
    array : array_like
    mean : array_like
    variance : array_like
    preserve_origin : bool, optional
        Whether to retain the origin (sign) of the data.
    """
    backend = get_backend(array)
    if preserve_origin:
        scale = backend.sqrt(variance + mean ** 2)
        return array * scale
    else:
        return mean + array * backend.sqrt(variance)


class Normalizer(object):
    """Normalizer Class."""

    def __init__(self, preserve_origin=False):
        super().__init__()
        self._mean = None
        self._variance = None
        self._count = 0
        self._preserve_origin = preserve_origin

    def __call__(self, array):
        return normalize(array, self._mean, self._variance, self._preserve_origin)

    def inverse(self, array):
        return denormalize(array, self._mean, self._variance, self._preserve_origin)

    def update(self, array):
        backend = get_backend(array)
        if self._mean is None:
            self._mean = backend.zeros(1)
        if self._variance is None:
            self._variance = backend.ones(1)

        new_mean = backend.mean(array, 0)
        new_var = backend.var(array, 0)
        new_count = len(array)

        self._variance = update_var(self._mean, self._variance, self._count,
                                    new_mean, new_var, new_count,
                                    backend is np)
        self._mean = update_mean(self._mean, self._count, new_mean, new_count)

        self._count += new_count
