"""MPC Algorithms."""
import torch
from torch.distributions import MultivariateNormal

from rllib.util.neural_networks.utilities import repeat_along_dimension
from rllib.util.rollout import rollout_actions
from rllib.dataset.utilities import stack_list_of_tuples
from rllib.util.utilities import discount_sum, sample_mean_and_cov


# TODO: ADD WARM STARTING through the mean.


def _eval_mpc(dynamical_model, reward_model, horizon, state, num_samples, gamma,
              action_sequence, termination=None, terminal_reward=None):

    trajectory = rollout_actions(dynamical_model, reward_model, action_sequence, state,
                                 termination)

    trajectory = stack_list_of_tuples(trajectory)
    returns = discount_sum(trajectory.reward, gamma)

    if terminal_reward:
        final_state = trajectory.next_state[-1]
        returns = returns + gamma ** horizon * terminal_reward(final_state)
    return returns


def random_shooting(dynamical_model, reward_model, horizon, x0, gamma=1,
                    num_samples=None, termination=None, terminal_reward=None,
                    warm_start=None):
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
    gamma: float.
        Discount factor.
    termination: Callable, optional.
        Termination condition.
    terminal_reward: terminal reward model, optional.

    Returns
    -------
    u: array of actions.
    """
    return cem_shooting(dynamical_model, reward_model, horizon, x0,
                        num_samples=num_samples, num_elites=1, num_iter=1,
                        gamma=gamma, termination=termination,
                        terminal_reward=terminal_reward, warm_start=warm_start)


def cem_shooting(dynamical_model, reward_model, horizon, x0, gamma=1, num_samples=None,
                 num_iter=5, num_elites=None, termination=None, terminal_reward=None,
                 warm_start=None):
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
    num_samples: int, optional.
        Number of samples for random shooting method.
    num_elites: int, optional.
        Number of elite samples to keep between iterations.
    num_iter: int, optional.
        Number of iterations of CEM method..
    gamma: float, optional.
        Discount factor.
    termination: Callable, optional.
        Termination condition.
    terminal_reward: terminal reward model, optional.

    Returns
    -------
    u: array of actions.
    """
    num_samples = 10 * horizon if not num_samples else num_samples
    num_elites = max(5, num_samples // 10) if not num_elites else num_elites

    state = x0

    dim_action = dynamical_model.dim_action
    if warm_start is not None:
        mean = warm_start
    else:
        mean = torch.zeros(*state.shape[:-1], horizon, dynamical_model.dim_action)

    covariance = 0.3 * torch.eye(dim_action).repeat(*state.shape[:-1], horizon, 1, 1)

    state = repeat_along_dimension(state, number=num_samples, dim=-2)

    for i in range(num_iter):
        action_distribution = MultivariateNormal(mean, covariance)
        action_sequence = action_distribution.sample((num_samples,))
        action_sequence = action_sequence.transpose(0, -2)

        returns = _eval_mpc(
            dynamical_model, reward_model, horizon, state, num_samples, gamma,
            action_sequence, termination=termination, terminal_reward=terminal_reward)

        idx = torch.topk(returns, k=num_elites, largest=True, dim=-1)[1]
        idx = idx.unsqueeze(0).unsqueeze(-1)  # Expand dims to action_sequence.
        idx = idx.repeat_interleave(horizon, 0).repeat_interleave(dim_action, 1)
        elite_actions = torch.gather(action_sequence, -2, idx)

        mean, covariance = sample_mean_and_cov(elite_actions.transpose(-1, -2))
        mean, covariance = mean.transpose(0, -2), covariance.transpose(0, -3)

    return mean
