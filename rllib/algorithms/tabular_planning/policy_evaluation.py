"""Policy Evaluation Algorithms."""

import numpy as np
import torch

from rllib.environment import mdp2mrp, transitions2kernelreward
from rllib.util.utilities import tensor_to_distribution

from .utilities import init_value_function


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

    kernel, reward = transitions2kernelreward(
        mrp.transitions, model.num_states, model.num_actions
    )

    A = torch.eye(model.num_states) - gamma * kernel[:, 0, :]
    # torch.testing.assert_allclose(A.inverse() @ A, torch.eye(model.num_states))
    vals = A.inverse() @ reward[:, 0]
    for state in range(model.num_states):
        value_function.set_value(state, vals[state].item())

    return value_function


def iterative_policy_evaluation(
    policy, model, gamma, eps=1e-6, max_iter=1000, value_function=None
):
    """Implement Policy Evaluation algorithm (policy iteration without max).

    Parameters
    ----------
    policy: AbstractPolicy
        policy to evaluate_agent.
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
        for state in range(model.num_states):
            if state in model.terminal_states:
                continue

            value_estimate = torch.tensor(0.0)
            state = torch.tensor(state).long()
            policy_ = tensor_to_distribution(policy(state), **policy.dist_params)
            for action in np.where(policy_.probs.detach().numpy())[0]:
                p_action = policy_.probs[action].item()
                for transition in model.transitions[(state.item(), action)]:
                    next_state = torch.tensor(transition["next_state"]).long()

                    value_estimate += (
                        p_action
                        * transition["probability"]
                        * (transition["reward"] + gamma * value_function(next_state))
                    )

            value = value_function(state)
            error = torch.abs(value_estimate - value).item()
            max_error = max(max_error, error)
            value_function.set_value(state, value_estimate)

        if max_error < eps:
            break

    return value_function
