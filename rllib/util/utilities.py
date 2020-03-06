"""Utilities for the rllib library."""

import scipy.signal
import torch
import torch.distributions
import numpy as np

__all__ = ['integrate', 'mellow_max', 'discount_cumsum', 'moving_average_filter',
           'separated_kl']


def integrate(function, distribution, num_samples=1):
    """Integrate a function over a distribution.

    Parameters
    ----------
    function: Callable.
        Function to integrate.
    distribution: Distribution.
        Distribution to integrate the function w.r.t..
    num_samples: int.
        Number of samples in MC integration.

    Returns
    -------
    integral value.
    """
    batch_size = distribution.batch_shape
    ans = torch.zeros(batch_size)
    if distribution.has_enumerate_support:
        for action in distribution.enumerate_support():
            prob = distribution.probs.gather(-1, action.unsqueeze(-1)).squeeze()
            q_val = function(action)
            ans += prob * q_val

    else:
        for _ in range(num_samples):
            q_val = function(distribution.rsample())
            ans += q_val

    return ans


def mellow_max(values, omega=1.):
    """Find mellow-max of an array of values.

    The mellow max is log(1/n sum(e^x)).

    Parameters
    ----------
    values: Array.
        array of values to find mellow max.
    omega: float, optional (default=1.).
        parameter of mellow-max.

    References
    ----------
    Asadi, Kavosh, and Michael L. Littman.
    "An alternative softmax operator for reinforcement learning."
    Proceedings of the 34th International Conference on Machine Learning. 2017.
    """
    n = torch.tensor(values.shape[-1]).float()
    return (torch.logsumexp(omega * values, dim=-1) - torch.log(n)) / omega


def discount_cumsum(rewards, gamma=1.0):
    """Get cumulative discounted returns.

    Given a vector [r0, r1, r2], return [r0 + gamma r1 + gamma^2 r2, r1 + gamma r2, r2].

    Parameters
    ----------
    rewards: array of rewards.
    gamma: discount factor.

    Returns
    -------
    discounted_returns: sum of discounted returns.

    References
    ----------
    From rllab.
    """
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


def mc_return(trajectory, gamma=1.0, bootstrap=None):
    """Calculate MC return from the trajectory."""
    if len(trajectory) == 0:
        return 0.
    discount = 1.
    value = torch.zeros_like(trajectory[0].reward)
    for observation in trajectory:
        value += discount * observation.reward
        discount *= gamma

    if bootstrap is not None:
        final_state = trajectory[-1].next_state
        is_terminal = trajectory[-1].done
        value += discount * bootstrap(final_state) * (1 - is_terminal)
    return value


def moving_average_filter(x, y, horizon: int):
    """Apply a moving average filter to data x and y.

    This function truncates the data to match the horizon.

    Parameters
    ----------
    x : ndarray
        The time stamps of the data.
    y : ndarray
        The values of the data.
    horizon : int
        The horizon over which to apply the filter.

    Returns
    -------
    x_smooth : ndarray
        A shorter array of x positions for smoothed values.
    y_smooth : ndarray
        The corresponding smoothed values
    """
    horizon = min(horizon, len(y))

    smoothing_weights = np.ones(horizon) / horizon
    x_smooth = x[horizon // 2: -horizon // 2 + 1]
    y_smooth = np.convolve(y, smoothing_weights, 'valid')
    return x_smooth, y_smooth


def separated_kl(p, q):
    """Compute the mean and variance components of the average KL divergence.

    Separately computes the mean and variance contributions to the KL divergence
    KL(p || q).

    Parameters
    ----------
    p : torch.distributions.MultivariateNormal
    q : torch.distributions.MultivariateNormal

    Returns
    -------
    kl_mean : torch.Tensor
    kl_var : torch.Tensor
    """
    p_mean, p_scale = p.loc, p.covariance_matrix
    q_mean, q_scale = q.loc, q.covariance_matrix

    pi_mean = torch.distributions.MultivariateNormal(p_mean, q_scale)
    pi_var = torch.distributions.MultivariateNormal(q_mean, p_scale)

    kl_mean = torch.distributions.kl_divergence(pi_mean, q).mean()
    kl_var = torch.distributions.kl_divergence(pi_var, q).mean()

    return kl_mean, kl_var
