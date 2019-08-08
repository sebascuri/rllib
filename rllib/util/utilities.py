"""Utilities for the rllib library."""


import numpy as np


__all__ = ['mc_value', 'sum_discounted_rewards']


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
