"""Value Iteration Algorithm."""

import torch

from rllib.algorithms.tabular_planning.utilities import init_value_function
from rllib.policy import TabularPolicy


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

            value_ = torch.zeros(model.num_actions)
            for action in range(model.num_actions):
                value_estimate = 0
                for transition in model.transitions[(state, action)]:
                    next_state = torch.tensor(transition["next_state"]).long()
                    value_estimate += transition["probability"] * (
                        transition["reward"] + gamma * value_function(next_state)
                    )
                value_[action] = value_estimate
            state = torch.tensor(state).long()
            value = value_function(state)
            value_, action = torch.max(value_, 0)

            error = max(error, torch.abs(value_ - value.item()))
            value_function.set_value(state, value_)
            policy.set_value(state, action)

        if error < eps:
            break

    return policy, value_function
