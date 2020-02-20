"""Utilities for the rllib library."""

import scipy.signal
import torch
import numpy as np

__all__ = ['integrate', 'mellow_max', 'discount_cumsum']


def integrate(function, distribution, num_samples=1):
    """Integrate a function over a distribution.

    Parameters
    ----------
    function: Callable.
        Function to integrate.
    distribution: Distribution.
        Distribution to integrate the function w.r.t..
    num_samples: int.
        Number of samples in MC integration.

    Returns
    -------
    integral value.
    """
    batch_size = distribution.batch_shape
    ans = torch.zeros(batch_size)
    if distribution.has_enumerate_support:
        for action in distribution.enumerate_support():
            prob = distribution.probs.gather(-1, action.unsqueeze(-1)).squeeze()
            q_val = function(action)
            ans += prob * q_val

    else:
        for _ in range(num_samples):
            q_val = function(distribution.rsample())
            ans += q_val

    return ans


def mellow_max(values, omega=1.):
    """Find mellow-max of an array of values.

    The mellow max is log(1/n sum(e^x)).

    Parameters
    ----------
    values: Array.
        array of values to find mellow max.
    omega: float, optional (default=1.).
        parameter of mellow-max.

    References
    ----------
    Asadi, Kavosh, and Michael L. Littman.
    "An alternative softmax operator for reinforcement learning."
    Proceedings of the 34th International Conference on Machine Learning. 2017.
    """
    n = torch.tensor(values.shape[-1]).float()
    return (torch.logsumexp(omega * values, dim=-1) - torch.log(n)) / omega


def discount_cumsum(rewards, gamma=1.0):
    """Get cumulative discounted returns.

    Given a vector [r0, r1, r2], return [r0 + gamma r1 + gamma^2 r2, r1 + gamma r2, r2].

    Parameters
    ----------
    rewards: array of rewards.
    gamma: discount factor.

    Returns
    -------
    discounted_returns: sum of discounted returns.

    References
    ----------
    From rllab.
    """
    if type(rewards) is np.ndarray:
        # The copy is for future transforms to pytorch
        return scipy.signal.lfilter([1], [1, -float(gamma)], rewards[::-1]
                                    )[::-1].copy()

    val = torch.zeros_like(rewards)
    r = 0
    for i, reward in enumerate(reversed(rewards)):
        r = reward + gamma * r
        val[-1 - i] = r
    return val
