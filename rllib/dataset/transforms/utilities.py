"""Utilities for the transformers."""
import torch
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal


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


def shift_mvn(mvn, mean, variance=None):
    """Shift a Multivariate Normal with a mean and a variance.

    MVNs from gpytorch do not admit
    """
    mu = mvn.mean
    sigma = mvn.covariance_matrix
    if not isinstance(mvn, MultitaskMultivariateNormal):
        if variance is None:
            variance = 1.0
        scale = torch.sqrt(variance)
        return MultivariateNormal(
            mu * scale + mean, covariance_matrix=sigma * scale ** 2
        )
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
        cov_ = sigma[
            ...,
            i * num_points : (i + 1) * num_points,
            i * num_points : (i + 1) * num_points,
        ]

        mvns.append(
            shift_mvn(MultivariateNormal(mean_, cov_), mean[..., i], variance[..., i])
        )
    return MultitaskMultivariateNormal.from_independent_mvns(mvns)
