"""Utilities for the transformers."""
from rllib.dataset.utilities import get_backend
import torch
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal


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
    if isinstance(array, MultitaskMultivariateNormal):
        return shift_mvn(array, mean, variance)

    backend = get_backend(array)
    if preserve_origin:
        scale = backend.sqrt(variance + mean ** 2)
        return array * scale
    else:
        return mean + array * backend.sqrt(variance)


def shift_mvn(mvn, mean, variance=None):
    """Shift a Multivariate Normal with a mean and a variance.

    MVNs from gpytorch do not admit
    """
    mu = mvn.mean
    sigma = mvn.covariance_matrix
    if not isinstance(mvn, MultitaskMultivariateNormal):
        if variance is None:
            variance = 1.
        scale = torch.sqrt(variance)
        return MultivariateNormal(mu * scale + mean,
                                  covariance_matrix=sigma * scale ** 2)
    if mvn.mean.dim() == 2:
        batch_size = None
        num_points, num_tasks = mvn.mean.shape
    else:
        batch_size, num_points, num_tasks = mvn.mean.shape

    if variance is None:
        variance = torch.ones(num_tasks)

    mvns = []
    for i in range(num_tasks):
        mean_ = mu[..., i]
        cov_ = sigma[...,
                     i * num_points:(i + 1) * num_points,
                     i * num_points:(i + 1) * num_points]

        mvns.append(shift_mvn(MultivariateNormal(mean_, cov_),
                              mean[..., i],
                              variance[..., i]))
    return MultitaskMultivariateNormal.from_independent_mvns(mvns)
