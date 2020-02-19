"""Utilities for the rllib library."""

import numpy as np
import torch

__all__ = ['mc_value', 'sum_discounted_rewards', 'integrate', 'mellow_max']


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


def _mc_value_slow(trajectory, gamma=1.0):
    """Monte-Carlo estimation of the value given a trajectory.

    Parameters
    ----------
    trajectory: sized list of Observations.
        list of Observations
    gamma: float, optional (default=1.0)
        discount factor

    Returns
    -------
    estimate: ndarray
        Monte-Carlo estimate of value_functions for the trajectory.
    """
    q_estimate = []
    for t in range(len(trajectory)):
        q_t = 0
        for i, observation in enumerate(trajectory[t:]):
            q_t = q_t + gamma ** i * observation.reward
        q_estimate.append(q_t)

    return np.array(q_estimate)


def mc_value(trajectory, gamma=1.0):
    """Monte-Carlo estimation of the value given a trajectory.

    Parameters
    ----------
    trajectory: sized list of Observations.
        list of Observations
    gamma: float, optional (default=1.0)
        discount factor

    Returns
    -------
    estimate: ndarray
        Monte-Carlo estimate of value_functions for the trajectory.
    """
    value_estimate = [0] * len(trajectory)
    value_estimate[-1] = trajectory[-1].reward

    for t in reversed(range(len(trajectory) - 1)):
        value_estimate[t] = trajectory[t].reward + gamma * value_estimate[t + 1]

    return np.array(value_estimate)


def sum_discounted_rewards(trajectory, gamma):
    """Sum of discounted rewards seen in a trajectory.

    Parameters
    ----------
    trajectory: sized list of Observations.
        list of Observations
    gamma: float
        discount factor

    Returns
    -------
    sum: float
    """
    rewards = []
    for observation in trajectory:
        rewards.append(observation.reward)
    rewards = np.array(rewards)
    i = np.arange(len(rewards))
    return np.sum(rewards * np.power(gamma, i))


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
