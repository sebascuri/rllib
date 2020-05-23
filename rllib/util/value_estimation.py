"""Utilities to estimate value functions."""
from collections import namedtuple

import numpy as np
import scipy
import scipy.signal
import torch

from rllib.util.utilities import RewardTransformer
from rllib.util.neural_networks.utilities import repeat_along_dimension
from rllib.util.rollout import rollout_model

MBValueReturn = namedtuple('MBValueReturn', ['value_estimate', 'trajectory'])


def discount_cumsum(rewards, gamma=1.0, reward_transformer=RewardTransformer()):
    r"""Get discounted cumulative sum of an array.

    Given a vector [r0, r1, r2], the discounted cum sum is another vector:
    .. math:: [r0 + gamma r1 + gamma^2 r2, r1 + gamma r2, r2].


    Parameters
    ----------
    rewards: Array.
        Array of rewards
    gamma: float, optional.
        Discount factor.
    reward_transformer: RewardTransformer, optional.

    Returns
    -------
    discounted_returns: Array.
        Sum of discounted returns.

    References
    ----------
    From rllab.
    """
    rewards = reward_transformer(rewards)
    if type(rewards) is np.ndarray:
        # The copy is for future transforms to pytorch
        return scipy.signal.lfilter([1], [1, -float(gamma)], rewards[::-1]
                                    )[::-1].copy()

    val = torch.zeros_like(rewards)
    r = 0
    for i, reward in enumerate(reversed(rewards)):
        r = reward + gamma * r
        val[-1 - i] = r
    return val


def discount_sum(rewards, gamma=1.0, reward_transformer=RewardTransformer()):
    r"""Get discounted sum of returns.

    Given a vector [r0, r1, r2], the discounted sum is tensor:
    .. math:: r0 + gamma r1 + gamma^2 r2

    Parameters
    ----------
    rewards: Tensor.
        Array of rewards. Either 1-d or 2-d. When 2-d, [trajectory x num_samples].
    gamma: float, optional.
        Discount factor.
    reward_transformer: RewardTransformer

    Returns
    -------
    cum_sum: tensor.
        Cumulative sum of returns.
    """
    rewards = reward_transformer(rewards)
    if rewards.dim() == 0:
        return rewards
    elif rewards.dim() == 1:
        steps = len(rewards)
        return (torch.pow(gamma * torch.ones(steps), torch.arange(steps)) * rewards
                ).sum()
    else:
        steps = rewards.shape[0]
        return torch.einsum('i,i...->...',
                            torch.pow(gamma * torch.ones(steps), torch.arange(steps)),
                            rewards)


def mc_return(trajectory, gamma=1.0, value_function=None, entropy_reg=0.,
              reward_transformer=RewardTransformer()):
    r"""Calculate n-step MC return from the trajectory.

    The N-step return of a trajectory is calculated as:
    .. math:: V(s) = \sum_{t=0}^T \gamma^t (r + \lambda H) + \gamma^{T+1} V(s_{T+1}).

    Parameters
    ----------
    trajectory: List[Observation]
        List of observations to compute the n-step return.
    gamma: float, optional.
        Discount factor.
    value_function: AbstractValueFunction, optional.
        Value function to bootstrap the value of the final state.
    entropy_reg: float, optional.
        Entropy regularization coefficient.
    reward_transformer: RewardTransformer

    """
    if len(trajectory) == 0:
        return 0.
    discount = 1.
    value = torch.zeros_like(trajectory[0].reward)
    for observation in trajectory:
        value += discount * (reward_transformer(observation.reward)
                             + entropy_reg * observation.entropy)
        discount *= gamma

    if value_function is not None:
        final_state = trajectory[-1].next_state
        is_terminal = trajectory[-1].done
        value += discount * value_function(final_state) * (1. - is_terminal)
    return value


def mb_return(state, dynamical_model, reward_model, policy, num_steps=1, gamma=1.,
              value_function=None, num_samples=1, entropy_reg=0.,
              reward_transformer=RewardTransformer(), termination=None):
    r"""Estimate the value of a state by propagating the state with a model for N-steps.

    Rolls out the model for a number of `steps` and sums up the rewards. After this,
    it bootstraps using the value function. With T = steps:

    .. math:: V(s) = \sum_{t=0}^T \gamma^t r(s, \pi(s)) + \gamma^{T+1} V(s_{T+1})

    Note that `steps=0` means that the model is still used to predict the next state.

    Parameters
    ----------
    state: torch.Tensor
        Initial state from which planning starts. It accepts a batch of initial states.
    dynamical_model: AbstractModel
        The model predicts a distribution over next states given states and actions.
    reward_model: AbstractReward
        The reward predicts a distribution over floats or ints given states and actions.
    policy: AbstractPolicy
        The policy predicts a distribution over actions given the state.
    num_steps: int, optional. (default=1).
        Number of steps predicted with the model before (optionally) bootstrapping.
    gamma: float, optional. (default=1.).
        Discount factor.
    value_function: AbstractValueFunction, optional. (default=None).
        The value function used for bootstrapping, takes states as input.
    num_samples: int, optional. (default=0).
        The states are repeated `num_repeats` times in order to estimate the expected
        value by MC sampling of the policy, rewards and dynamics (jointly).
    entropy_reg: float, optional. (default=0).
        Entropy regularization parameter.
    termination: Callable, optional. (default=None).
        Callable that returns True if the transition yields a terminal state.
    reward_transformer: RewardTransformer.

    Returns
    -------
    return: DynaReturn
        q_target:
            Num_samples of MC estimation of q-function target.
        trajectory:
            Sample trajectory that MC estimation produces.

    References
    ----------
    Lowrey, K., Rajeswaran, A., Kakade, S., Todorov, E., & Mordatch, I. (2018).
    Plan online, learn offline: Efficient learning and exploration via model-based
    control. ICLR.

    Sutton, R. S. (1991).
    Dyna, an integrated architecture for learning, planning, and reacting. ACM.

    Silver, D., Sutton, R. S., & Müller, M. (2008).
    Sample-based learning and search with permanent and transient memories. ICML.
    """
    # Repeat states to get a better estimate of the expected value
    state = repeat_along_dimension(state, number=num_samples, dim=0)
    trajectory = rollout_model(dynamical_model, reward_model, policy, state,
                               max_steps=num_steps, termination=termination)
    value = mc_return(trajectory, gamma=gamma, value_function=value_function,
                      entropy_reg=entropy_reg, reward_transformer=reward_transformer)

    return MBValueReturn(value, trajectory)
