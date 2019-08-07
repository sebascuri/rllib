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

    for t in reversed(range(1, len(trajectory))):
        value_estimate[t-1] = trajectory[t].reward + gamma * value_estimate[t]

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


# if __name__ == "__main__":
#     from rllib.dataset import Observation
#     trajectory = []
#     for i in range(10):
#         observation = Observation(state=0, action=1, next_state=2, reward=1, done=0)
#         trajectory.append(observation)
#
#     gamma = 0.9
#     print(mc_value(trajectory, gamma),
#           mc_value_slow(trajectory, gamma),
#           sum_discounted_rewards(trajectory, gamma)
#           )
