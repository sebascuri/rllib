"""MPC Algorithms."""
import torch
from torch.distributions import MultivariateNormal

from rllib.util.neural_networks.utilities import repeat_along_dimension
from rllib.util.rollout import rollout_actions
from rllib.dataset.utilities import stack_list_of_tuples
from rllib.util.utilities import discount_sum, sample_mean_and_cov


def _eval_mpc(dynamical_model, reward_model, horizon, state, num_samples, gamma,
              action_distribution, terminal_reward=None):

    action_sequence = action_distribution.sample((num_samples,))
    action_sequence = action_sequence.transpose(0, 1)

    trajectory = rollout_actions(dynamical_model, reward_model, action_sequence, state,
                                 max_steps=horizon)

    trajectory = stack_list_of_tuples(trajectory)
    returns = discount_sum(trajectory.reward, gamma)

    if terminal_reward:
        final_state = trajectory.next_state[-1]
        returns = returns + gamma ** horizon * terminal_reward(final_state)
    return action_sequence[:, torch.argsort(returns, descending=True), :]


def random_shooting(dynamical_model, reward_model, horizon, x0,
                    num_samples, gamma=1, terminal_reward=None):
    r"""Solve the discrete time mpc controller with Random Shooting.

    ..math :: u[0:H-1] = \arg \max \sum_{t=0}^{H-1} r(x0, u) + final_reward(x_H)

    Random Shooting solves the MPC problem by randomly sampling a sequence of actions
    and returning the sequence with higher value.

    Parameters
    ----------
    dynamical_model: state transition model.
    reward_model: reward model.
    horizon: int.
        Horizon to solve planning problem..
    x0: initial state.
    num_samples: int.
        Number of samples for random shooting method.
    gamma: discount factor.
    terminal_reward: terminal reward model, optional.

    Returns
    -------
    u: array of actions.
    """
    return cem_shooting(dynamical_model, reward_model, horizon, x0, num_samples,
                        num_elites=1, num_iter=1, gamma=gamma,
                        terminal_reward=terminal_reward)


def cem_shooting(dynamical_model, reward_model, horizon, x0, num_samples, num_elites,
                 num_iter, gamma=1, terminal_reward=None):
    r"""Solve the discrete time mpc controller with Random Shooting.

    ..math :: u[0:H-1] = \arg \max \sum_{t=0}^{H-1} r(x0, u) + final_reward(x_H)

    CEM solves the MPC problem by adaptively sampling a sequence of actions.
    The sampling distribution is adapted by fitting a Multivariate Gaussian to the
    best `num_elite' action sequences for `num_iter' times.

    Parameters
    ----------
    dynamical_model: state transition model.
    reward_model: reward model.
    horizon: int.
        Horizon to solve planning problem..
    x0: initial state.
    num_samples: int.
        Number of samples for random shooting method.
    num_elites: int.
        Number of elite samples to keep between iterations.
    num_iter: int.
        Number of iterations of CEM method..
    gamma: discount factor.
    terminal_reward: terminal reward model, optional.

    Returns
    -------
    u: array of actions.
    """
    state = x0
    if num_samples > 1:
        state = repeat_along_dimension(state, number=num_samples, dim=0)

    dim_action = dynamical_model.dim_action
    mean = torch.zeros(horizon, dim_action)
    covariance = torch.eye(dim_action).repeat(horizon, 1, 1)

    for i in range(num_iter):
        action_distribution = MultivariateNormal(mean, covariance)
        action_sequence = _eval_mpc(dynamical_model, reward_model, horizon, state,
                                    num_samples, gamma, action_distribution,
                                    terminal_reward=terminal_reward)

        elite_actions = action_sequence[:, :num_elites, :]
        mean, covariance = sample_mean_and_cov(elite_actions.transpose(-1, -2))

    return mean
