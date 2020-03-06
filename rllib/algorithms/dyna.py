"""Implementation of Dyna algorithm."""

from rllib.util.neural_networks.utilities import repeat_along_dimension
from rllib.util.rollout import rollout_model
from rllib.util.utilities import mc_return
from collections import namedtuple

DynaReturn = namedtuple('DynaReturn', ['q_target', 'trajectory'])


def dyna_rollout(state, model, policy, reward, steps, gamma=0.99, value_function=None,
                 num_samples=1):
    r"""Estimate the value of a system with the model and the value function.

    Rolls out the model for a number of `steps` and sums up the rewards. After this,
    it bootstraps using the value function. With T = steps:

    .. math:: V(s) = \sum_{t=0}^T \gamma^t r(s, \pi(s)) + \gamma^{T+1} V(s_{T+1})

    Note that `steps=0` means that the model is still used to predict the next state.

    Parameters
    ----------
    state : torch.Tensor
        Initial state from which planning starts. It accepts a batch of initial states.
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
    value_function : AbstractValueFunction
        The value function used for bootstrapping, takes states as input.
    num_samples : int
        If great than 1, the states are repeated `num_repeats` times in order to
        estimate the expected performance by MC sampling.

    Returns
    -------
    return : DynaReturn
        q_target: num_samples of MC estimation of q-function target.
        trajectory: sample trajectory that MC estimation produces.
    """
    # Repeat states to get a better estimate of the expected value
    if num_samples > 1:
        state = repeat_along_dimension(state, number=num_samples, dim=0)
    trajectory = rollout_model(model, reward, policy, state, max_steps=steps + 1)
    value = mc_return(trajectory, gamma=gamma, value_function=value_function)

    return DynaReturn(value, trajectory)
