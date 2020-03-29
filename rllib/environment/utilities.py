"""Utilities for environment module."""

from collections import defaultdict
from itertools import product

import numpy as np
import torch

from rllib.util.utilities import tensor_to_distribution
from rllib.environment.mdp import MDP


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

    kernel = np.zeros((num_states, num_actions, num_states))
    reward = np.zeros((num_states, num_actions))

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

    return MDP(kernelreward2transitions(kernel, reward),
               num_states, num_actions,
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
    mrp_kernel = np.zeros((environment.num_states, 1, environment.num_states))
    mrp_reward = np.zeros((environment.num_states, 1))

    for state in range(environment.num_states):
        if state in environment.terminal_states:
            mrp_kernel[state, 0, state] = 1
            mrp_reward[state] = 0
            continue

        state = torch.tensor(state).long()
        policy_ = tensor_to_distribution(policy(state))

        for a, p_action in enumerate(policy_.probs):
            for transition in environment.transitions[(state.item(), a)]:
                p_ns = transition['probability']
                mrp_reward[state, 0] += p_action * p_ns * transition['reward']
                mrp_kernel[state, 0, transition['next_state']] += p_action * p_ns

    return MDP(kernelreward2transitions(mrp_kernel, mrp_reward),
               environment.num_states, 1,
               initial_state=environment.initial_state,
               terminal_states=environment.terminal_states)


def transitions2kernelreward(transitions, num_states, num_actions):
    """Transform a dictionary of transitions to kernel, reward matrices."""
    kernel = np.zeros((num_states, num_actions, num_states))
    reward = np.zeros((num_states, num_actions))
    for (state, action), transition in transitions.items():
        for data in transition:
            kernel[state, action, data['next_state']] = data['probability']
            reward[state, action] += data['reward'] * data['probability']

    return kernel, reward


def kernelreward2transitions(kernel, reward):
    """Transform a kernel and reward matrix into a transition dicitionary."""
    transitions = defaultdict(list)

    num_states, num_actions = reward.shape

    for state in range(num_states):
        for action in range(num_actions):
            for next_state in np.where(kernel[state, action])[0]:
                transitions[(state, action)].append(
                    {'next_state': next_state,
                     'probability': kernel[state, action, next_state],
                     'reward': reward[state, action]}
                )

    return transitions
