"""Utilities for the rllib library."""

import numpy as np
import scipy.signal
import torch
import torch.distributions
from .distributions import Delta
from torch.distributions import Categorical, MultivariateNormal


def get_backend(array):
    """Get backend of the array."""
    if isinstance(array, torch.Tensor):
        return torch
    elif isinstance(array, np.ndarray):
        return np
    else:
        raise TypeError


def integrate(function, distribution, num_samples=1):
    r"""Integrate a function over a distribution.

    Compute:
    .. math:: \int_a function(a) distribution(a) da.

    When the distribution is discrete, just sum over the actions.
    When the distribution is continuous, approximate the integral via MC sampling.

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
            f_val = function(action)
            ans += prob.detach() * f_val

    else:
        for _ in range(num_samples):
            if distribution.has_rsample:
                action = distribution.rsample()
            else:
                action = distribution.sample()
            ans += function(action).squeeze() / num_samples
    return ans


def mellow_max(values, omega=1.):
    r"""Find mellow-max of an array of values.

    The mellow max is:
    .. math:: mm_\omega(x) =1 / \omega \log(1 / n \sum_{i=1}^n e^{\omega x_i}).

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
    n = torch.tensor(values.shape[-1], dtype=torch.get_default_dtype())
    return (torch.logsumexp(omega * values, dim=-1) - torch.log(n)) / omega


def discount_cumsum(rewards, gamma=1.0):
    r"""Get discounted cumulative sum of an array.

    Given a vector [r0, r1, r2], the discounted cum sum is another vector:
    .. math:: [r0 + gamma r1 + gamma^2 r2, r1 + gamma r2, r2].


    Parameters
    ----------
    rewards: Array.
        Array of rewards
    gamma: float, optional.
        Discount factor.

    Returns
    -------
    discounted_returns: Array.
        Sum of discounted returns.

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


def discount_sum(rewards, gamma=1.0):
    r"""Get discounted sum of returns.

    Given a vector [r0, r1, r2], the discounted sum is tensor:
    .. math:: r0 + gamma r1 + gamma^2 r2

    Parameters
    ----------
    rewards: Tensor.
        Array of rewards. Either 1-d or 2-d. When 2-d, [trajectory x num_samples].
    gamma: float, optional.
        Discount factor.

    Returns
    -------
    cum_sum: tensor.
        Cumulative sum of returns.
    """
    if rewards.dim() == 0:
        return rewards
    elif rewards.dim() == 1:
        steps = len(rewards)
        return (torch.pow(gamma * torch.ones(steps), torch.arange(steps)) * rewards
                ).sum()
    else:
        steps = rewards.shape[0]
        return torch.pow(gamma * torch.ones(steps), torch.arange(steps)) @ rewards


def mc_return(trajectory, gamma=1.0, value_function=None, entropy_reg=0.):
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
    """
    if len(trajectory) == 0:
        return 0.
    discount = 1.
    value = torch.zeros_like(trajectory[0].reward)
    for observation in trajectory:
        value += discount * (observation.reward + entropy_reg * observation.entropy)
        discount *= gamma

    if value_function is not None:
        final_state = trajectory[-1].next_state
        is_terminal = trajectory[-1].done
        value += discount * value_function(final_state) * (1 - is_terminal)
    return value


def tensor_to_distribution(args):
    """Convert tensors to a distribution.

    When args is a tensor, it returns a Categorical distribution with logits given by
    args.

    When args is a tuple, it returns a MultivariateNormal distribution with args[0] as
    mean and args[1] as scale_tril matrix. When args[1] is zero, it returns a Delta.

    Parameters
    ----------
    args: Union[Tuple[Tensor], Tensor].
        Tensors with the parameters of a distribution.
    """
    if not isinstance(args, tuple):
        return Categorical(logits=args)
    elif torch.all(args[1] == 0):
        return Delta(args[0], event_dim=args[0].dim() - 1)
    else:
        return MultivariateNormal(args[0], scale_tril=args[1])


def separated_kl(p, q):
    """Compute the mean and variance components of the average KL divergence.

    Separately computes the mean and variance contributions to the KL divergence
    KL(p || q).

    Parameters
    ----------
    p: torch.distributions.MultivariateNormal
    q: torch.distributions.MultivariateNormal

    Returns
    -------
    kl_mean: torch.Tensor
    kl_var: torch.Tensor
    """
    p_mean = torch.distributions.MultivariateNormal(q.loc, scale_tril=p.scale_tril)
    p_var = torch.distributions.MultivariateNormal(p.loc, scale_tril=q.scale_tril)

    kl_mean = torch.distributions.kl_divergence(p=p_mean, q=q).mean()
    kl_var = torch.distributions.kl_divergence(p=p_var, q=q).mean()

    return kl_mean, kl_var


def sample_mean_and_cov(sample):
    """Compute mean and covariance of a sample of vectors.

    Parameters
    ----------
    sample: Tensor
        Tensor of dimensions [batch x N x num_samples].

    Returns
    -------
    mean: Tensor
        Tensor of dimension [batch x N].
    covariance: Tensor
        Tensor of dimension [batch x N x N].

    """
    num_samples = sample.shape[-1]
    mean = torch.mean(sample, dim=-1, keepdim=True)
    sigma = (mean - sample) @ (mean - sample).transpose(-2, -1)
    sigma += 1e-6 * torch.eye(sigma.shape[-1])  # Add some jitter.
    covariance = sigma / num_samples
    mean = mean.squeeze(-1)

    return mean, covariance
