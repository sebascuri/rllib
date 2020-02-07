"""Policy Evaluation Algorithms."""

from .utilities import init_value_function
from rllib.environment import mdp2mrp
import torch
import numpy as np


def linear_system_policy_evaluation(policy, model, gamma, value_function=None):
    """Evaluate a policy in an MDP solving the system bellman of equations.

    V = r + gamma * P * V
    V = (I - gamma * P)^-1 r
    """
    if model.num_actions is None or model.num_states is None:
        raise NotImplementedError("Actions and States must be discrete and countable.")

    if value_function is None:
        value_function = init_value_function(model.num_states, model.terminal_states)

    mrp = mdp2mrp(environment=model, policy=policy)
    A = torch.eye(model.num_states) - gamma * mrp.kernel[:, 0, :]
    # torch.testing.assert_allclose(A.inverse() @ A, torch.eye(model.num_states))
    vals = A.inverse() @ mrp.reward[:, 0]
    for state in range(model.num_states):
        value_function.set_value(state, vals[state].item())

    return value_function


def iterative_policy_evaluation(policy, model, gamma, eps=1e-6, max_iter=1000,
                                value_function=None):
    """Implement Policy Evaluation algorithm (policy iteration without max).

    Parameters
    ----------
    policy: AbstractPolicy
        policy to evaluate.
    model: AbstractModel
        a model of the environment. (also an MDP environment can be used).
    gamma: float
        discount factor.
    eps: float, optional
        desired precision, by default is 1e-6.
    max_iter: int, optional
        maximum number of iterations.
    value_function: TabularValueFunction, optional
        initial estimate of value function.

    Returns
    -------
    value_function: TabularValueFunction
        value function associated with the policy.


    References
    ----------
    Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction.
    MIT press.
    Chapter 4.1

    """
    if model.num_actions is None or model.num_states is None:
        raise NotImplementedError("Actions and States must be discrete and countable.")

    if value_function is None:
        value_function = init_value_function(model.num_states, model.terminal_states)

    for _ in range(max_iter):
        max_error = 0
        avg_error = 0
        for state_ in range(model.num_states):
            if state_ in model.terminal_states:
                continue
            state = torch.tensor(state_).long()

            value = value_function(state)
            value_estimate = torch.tensor(0.)
            policy_ = policy(state)
            for action in np.where(policy_.probs.detach().numpy())[0]:
                p_action = policy_.probs[action].item()
                value_estimate += p_action * model.reward[state, action]
                for next_state in np.where(model.kernel[state, action])[0]:
                    p_next = model.kernel[state, action, next_state]
                    next_state = torch.tensor(next_state).long()
                    next_val = value_function(next_state)
                    value_estimate += gamma * p_action * p_next * next_val

            error = torch.abs(value_estimate - value).item()
            if error > 1:
                print(state, error)
            max_error = max(max_error, error)
            avg_error += error
            value_function.set_value(state, value_estimate)

        if max_error < eps:
            break
        print(max_error, avg_error / (model.num_states - len(model.terminal_states)))

    return value_function
