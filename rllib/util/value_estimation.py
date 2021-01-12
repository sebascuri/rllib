"""Utilities to estimate value functions."""
from collections import namedtuple

import numpy as np
import scipy
import scipy.signal
import torch

from rllib.dataset.utilities import stack_list_of_tuples
from rllib.util.neural_networks.utilities import repeat_along_dimension
from rllib.util.rollout import rollout_model
from rllib.util.utilities import RewardTransformer, get_backend

MBValueReturn = namedtuple("MBValueReturn", ["value_estimate", "trajectory"])


def reward_to_go(
    rewards, gamma=1.0, reward_transformer=RewardTransformer(), terminal_reward=None
):
    """Compute rewards to go."""
    rewards = reward_transformer(rewards)
    n_steps = rewards.shape[-1]
    discount = torch.pow(torch.tensor(gamma), torch.arange(n_steps))

    discounted_sum_rewards = torch.nn.functional.conv1d(
        rewards.unsqueeze(1), discount.unsqueeze(0).unsqueeze(1), padding=n_steps
    )[:, 0, -(n_steps + 1) : -1]

    if terminal_reward is not None:
        discounted_sum_rewards += gamma ** n_steps * terminal_reward
    return discounted_sum_rewards


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
    bk = get_backend(rewards)
    if bk is torch and not rewards.requires_grad:
        rewards = rewards.numpy()
    if type(rewards) is np.ndarray:
        returns = scipy.signal.lfilter([1], [1, -gamma], rewards[..., ::-1])[..., ::-1]
        returns = returns.copy()  # The copy is for future transforms to pytorch

        if bk is torch:
            return torch.tensor(returns, dtype=torch.get_default_dtype())
        else:
            return returns

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
        return (
            torch.pow(gamma * torch.ones(steps), torch.arange(steps)) * rewards
        ).sum()
    else:
        steps = rewards.shape[1]
        return torch.einsum(
            "i,ki...->k...",
            torch.pow(gamma * torch.ones(steps), torch.arange(steps)),
            rewards,
        )


def n_step_return(
    observation,
    gamma=1.0,
    entropy_regularization=0.0,
    reward_transformer=RewardTransformer(),
    value_function=None,
    reduction="none",
):
    """Calculate all n-step returns for the observation.

    It expects an observation with shape batch x n-step x dim.
    It returns a tensor with shape batch x n-step.
    """
    while observation.reward.ndim < 2:
        observation.reward = observation.reward.unsqueeze(0)
    n_steps = observation.reward.shape[-1]
    discount = torch.pow(torch.tensor(gamma), torch.arange(n_steps))
    rewards = (
        reward_transformer(observation.reward)
        + entropy_regularization * observation.entropy
    )
    discounted_rewards = rewards * discount
    value = torch.cumsum(discounted_rewards, dim=-1)

    if value_function is not None:
        final_value = value_function(observation.next_state)
        not_done = 1.0 - observation.done
        if final_value.ndim > value.ndim:
            if reduction == "min":
                final_value = final_value.min(-1)[0]
            elif reduction == "mean":
                final_value = final_value.mean(-1)
            elif reduction == "none":
                num_q = final_value.shape[-1]
                value = value.unsqueeze(-1).repeat_interleave(
                    final_value.shape[-1], dim=-1
                )
                discount = discount.unsqueeze(-1).repeat_interleave(num_q, dim=-1)
                not_done = not_done.unsqueeze(-1).repeat_interleave(num_q, dim=-1)
            else:
                raise NotImplementedError(f"{reduction} not implemented.")
        value += gamma * discount * final_value * not_done
    return value


def mc_return(
    observation,
    gamma=1.0,
    lambda_=1.0,
    reward_transformer=RewardTransformer(),
    value_function=None,
    reduction="none",
    entropy_regularization=0.0,
):
    r"""Calculate n-step MC return from the trajectory.

    The N-step return of a trajectory is calculated as:
    .. math:: V(s) = \sum_{t=0}^T \gamma^t (r + \lambda H) + \gamma^{T+1} V(s_{T+1}).

    Parameters
    ----------
    observation: Observation
        List of observations to compute the n-step return.
    gamma: float, optional.
        Discount factor.
    lambda_: float, optional.
        Lambda return.
    value_function: AbstractValueFunction, optional.
        Value function to bootstrap the value of the final state.
    entropy_regularization: float, optional.
        Entropy regularization coefficient.
    reward_transformer: RewardTransformer
    reduction: str.
        How to reduce ensemble value functions.

    """
    if observation.reward.ndim == 0 or len(observation.reward) == 0:
        return 0.0
    returns = n_step_return(
        observation,
        gamma=gamma,
        reward_transformer=reward_transformer,
        entropy_regularization=entropy_regularization,
        value_function=value_function,
        reduction=reduction,
    )
    steps = returns.shape[1]  # Batch x T x num_q
    if steps == 1 or lambda_ == 1.0:
        if returns.ndim == 2:
            return returns[:, -1]
        else:
            return returns[..., -1, :]
    else:
        w = torch.cat(
            (
                (1 - lambda_) * lambda_ ** torch.arange(steps - 1),
                torch.tensor([lambda_]) ** (steps - 1),
            )
        )
        w = w.unsqueeze(0)
        if returns.ndim == 2:
            return (w * returns).sum(-1)
        else:
            return (w.unsqueeze(-1) * returns).sum(-2)


def mb_return(
    state,
    dynamical_model,
    reward_model,
    policy,
    num_steps=1,
    gamma=1.0,
    value_function=None,
    num_samples=1,
    entropy_reg=0.0,
    reward_transformer=RewardTransformer(),
    termination_model=None,
    reduction="none",
):
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
    termination_model: AbstractModel, optional. (default=None).
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
    trajectory = rollout_model(
        dynamical_model=dynamical_model,
        reward_model=reward_model,
        policy=policy,
        initial_state=state,
        max_steps=num_steps,
        termination_model=termination_model,
    )
    observation = stack_list_of_tuples(trajectory, dim=state.ndim - 1)
    value = mc_return(
        observation=observation,
        gamma=gamma,
        value_function=value_function,
        entropy_regularization=entropy_reg,
        reward_transformer=reward_transformer,
        reduction=reduction,
    )

    return MBValueReturn(value, observation)
