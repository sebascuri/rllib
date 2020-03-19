"""Value Iteration Algorithm."""

import numpy as np
import torch

from rllib.policy import TabularPolicy
from .utilities import init_value_function


def value_iteration(model, gamma, eps=1e-6, max_iter=1000, value_function=None):
    """Implement of Value Iteration algorithm.

    Parameters
    ----------
    model:
    gamma: discount factor.
    eps: desired precision of policy evaluation step
    max_iter: maximum number of iterations
    value_function: initial estimate of value function, optional.

    Returns
    -------
    policy:
    value_function:

    References
    ----------
    Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction.
    MIT press.
    Chapter 4.4

    """
    if model.num_actions is None or model.num_states is None:
        raise NotImplementedError("Actions and States must be discrete and countable.")

    if value_function is None:
        value_function = init_value_function(model.num_states, model.terminal_states)
    policy = TabularPolicy(num_states=model.num_states, num_actions=model.num_actions)

    for _ in range(max_iter):
        error = 0
        for state in range(model.num_states):
            state = torch.tensor(state).long()

            value = value_function(state)

            value_ = torch.zeros(model.num_actions)
            for action in range(model.num_actions):
                value_estimate = 0
                for next_state in np.where(model.kernel[state, action])[0]:
                    next_state = torch.tensor(next_state).long()
                    value_estimate += model.kernel[state, action, next_state] * (
                            model.reward[state, action]
                            + gamma * value_function(next_state)
                    )

                value_[action] = value_estimate

            value_, action = torch.max(value_, 0)
            error = max(error, torch.abs(value_ - value.item()))
            value_function.set_value(state, value_)
            policy.set_value(state, action)

        if error < eps:
            break

    return policy, value_function
