"""Utilities for environment module."""

from .mdp import MDP
import numpy as np
import torch
from itertools import product
from rllib.util.utilities import tensor_to_distribution


def gym2mdp(environment):
    """Transform discrete gym environment to an mdp.

    Parameters
    ----------
    environment: GymEnvironment.

    Returns
    -------
    environment: MDP.
    """
    num_states = environment.num_states
    num_actions = environment.num_actions
    transitions = environment.env.P

    kernel = torch.zeros((num_states, num_actions, num_states))
    reward = torch.zeros((num_states, num_actions))

    for state, action in product(range(num_states), range(num_actions)):
        if state == (num_states - 1):  # terminal state
            kernel[state, action, state] = 1
            reward[state, action] = 0
            continue
        for (p, ns, r, done) in transitions[state][action]:
            if done:
                ns = num_states - 1
            kernel[state, action, ns] = p
            reward[state, action] = r

    terminal_states = [num_states - 1]

    # Verify for correctness of kernel.
    for state, action in product(range(num_states), range(num_actions)):
        assert kernel[state, action].sum() == 1

    return MDP(transition_kernel=kernel, reward=reward,
               initial_state=environment.env.reset,
               terminal_states=terminal_states)


def mdp2mrp(environment, policy):
    """Transform MDP and Policy to an MRP.

    Parameters
    ----------
    environment: MDP.
    policy: AbstractPolicy.

    Returns
    -------
    environment: MDP.
    """
    mrp_kernel = torch.zeros((environment.num_states, 1, environment.num_states))
    mrp_reward = torch.zeros(environment.num_states, 1)

    for state in range(environment.num_states):
        if state in environment.terminal_states:
            mrp_kernel[state, 0, state] = 1
            mrp_reward[state] = 0
            continue

        state = torch.tensor(state).long()
        policy_ = tensor_to_distribution(policy(state))

        for a, p_action in enumerate(policy_.probs):
            mrp_reward[state, 0] += p_action * environment.reward[state, a]
            for next_state in np.where(environment.kernel[state, a])[0]:
                p_next_state = environment.kernel[state, a, next_state]
                mrp_kernel[state, 0, next_state] += p_action * p_next_state

    return MDP(mrp_kernel, mrp_reward,
               initial_state=environment.initial_state,
               terminal_states=environment.terminal_states)
