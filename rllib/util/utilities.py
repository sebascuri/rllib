"""Utilities for the rllib library."""

import numpy as np
import torch
from torch.distributions import constraints
from torch.distributions import Distribution
__all__ = ['mc_value', 'sum_discounted_rewards', 'Delta']


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

    for t in reversed(range(len(trajectory)-1)):
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


class Delta(Distribution):
    """Implementation of a Dirac Delta distribution."""

    arg_constraints = {'loc': constraints.real_vector}
    support = constraints.real  # type: ignore
    has_rsample = True

    def __init__(self, loc, validate_args=False):
        self.loc = loc
        batch_shape, event_shape = self.loc.shape[:-1], self.loc.shape[-1:]
        super().__init__(batch_shape=batch_shape, event_shape=event_shape,
                         validate_args=validate_args)

    @property
    def mean(self):
        """Return mean of distribution."""
        return self.loc

    @property
    def variance(self):
        """Return variance of distribution."""
        return torch.zeros_like(self.loc)

    def rsample(self, sample_shape=torch.Size()):
        """Return differentiable sample of distribution."""
        return self.loc.expand(sample_shape + self.loc.shape)

    def log_prob(self, value):
        """Return log probability of distribution."""
        if value == self.loc:
            return 0.
        else:
            return float('-Inf')

    def entropy(self):
        """Return entropy of distribution."""
        return float('-inf')
