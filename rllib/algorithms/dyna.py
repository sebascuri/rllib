"""Implementation of Dyna algorithm."""

from rllib.util.neural_networks.utilities import repeat_along_dimension
from rllib.dataset.datatypes import Observation
import torch
from collections import namedtuple

DynaReturn = namedtuple('DynaReturn', ['q_target', 'trajectory'])


def dyna_estimate(state, model, policy, reward, steps, gamma=0.99, bootstrap=None,
                  num_samples=1):
    r"""Estimate the value of a system with the model and the value function.

    Rolls out the model for a number of `steps` and sums up the rewards. After this,
    it bootstraps using the value function. With T = steps:

    .. math:: V(s) = \sum_{t=0}^T \gamma^t r(s, \pi(s)) + \gamma^{T+1} V(s_{T+1})

    Note that `steps=0` means that the model is still used to predict the next state.

    Parameters
    ----------
    state : torch.Tensor
        Initial state from which planning starts.
    model :
        The model predicts a distribution over next states given states and actions.
    policy :
        The policy predicts a distribution over actions given the state.
    reward :
        The reward predicts a distribution over floats or ints given states and actions.
    steps : int
        Number of steps predicted with the model before (optionally) bootstrapping.
    gamma : float, optional
        Discount factor.
    bootstrap : torch.nn.Module
        The value function used for bootstrapping, takes states as input.
    num_samples : int
        If great than 1, the states are repeated `num_repeats` times in order to
        estimate the expected performance by MC sampling.

    Returns
    -------
    values : torch.Tensor
        Has the same shape as `states` up to the last dimension
    """
    if state.dim() == 1:
        value = torch.zeros(num_samples)
    else:
        value = torch.zeros(num_samples, state.shape[-2])
    discount = 1.
    trajectory = []

    # Repeat states to get a better estimate of the expected value
    if num_samples > 1:
        state = repeat_along_dimension(state, number=num_samples, dim=0)

    for _ in range(steps + 1):
        # Sample an action
        action_distribution = policy(state)
        if action_distribution.has_rsample:
            action = action_distribution.rsample()
        else:
            action = action_distribution.sample()

        # % Sample a reward
        reward_distribution = reward(state, action)
        if reward_distribution.has_rsample:
            reward = reward_distribution.rsample()
        else:
            reward = reward_distribution.sample()

        value += discount * reward

        # Sample the next state
        next_state_distribution = model(state, action)
        if next_state_distribution.has_rsample:
            next_state = next_state_distribution.rsample()
        else:
            next_state = next_state_distribution.sample()

        trajectory.append(Observation(state, action, reward, next_state))

        # Update state and discount factor.
        state = next_state
        discount *= gamma

    # Bootstrap with the value function from the final states
    if bootstrap is not None:
        value += discount * bootstrap(state)

    return DynaReturn(value, trajectory)
