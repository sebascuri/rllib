"""Utilities for the model sub-module."""

from rllib.util.neural_networks.utilities import repeat_along_dimension


def estimate_value(states, model, policy, reward, steps, gamma=0.99, bootstrap=None,
                   num_samples=1):
    r"""Estimate the value of a system with the model and the value function.

    Rolls out the model for a number of `steps` and sums up the rewards. After this,
    it bootstraps using the value function. With T = num_steps:

    .. math:: V(s) = \sum_{t=0}^T \gamma^t r(s, \pi(s)) + \gamma^{T+1} V(s_{T+1})

    Note that `steps=0` means that the model is still used to predict the next state.

    Parameters
    ----------
    states : torch.tensor
    model : torch.nn.Module
        The model predicts a distribution over next states given states and actions.
    policy : torch.nn.Module
    reward : torch.nn.Module
        The reward function depends on states and actions.
    steps : int
        Number of steps predicted with the model before (optionally) bootstrapping.
    gamma : float, optional
    bootstrap : torch.nn.Module
        The value function used for bootstrapping, takes states as input.
    num_samples : int
        If great than 1, the states are repeated `num_repeats` times in order to
        estimate the expected performance.

    Returns
    -------
    values : torch.Tensor
        Has the same shape as `states` up to the last dimension
    """
    value = 0.
    discount = 1.

    # Repeat states to get a better estimate of the expected value
    if num_samples > 1:
        states = repeat_along_dimension(states, number=num_samples, dim=-2)

    for _ in range(steps + 1):
        actions = policy(states)
        try:
            if actions.has_rsample:
                actions = actions.rsample()
            else:
                actions = actions.sample()
        except AttributeError:
            pass

        value = value + discount * reward(states, actions)
        states = model(states, actions).rsample()
        # if states.has_rsample:
        #     states = states.rsample()
        # else:
        #     states = states.sample()
        discount *= gamma

    # Bootstrap with the value function from the final states
    if bootstrap is not None:
        value = value + discount * bootstrap(states)

    # Average out the additional samples
    if num_samples > 1:
        value = value.mean(dim=-2)

    return value
