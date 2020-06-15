"""Utilities for the rllib library."""
import warnings

import numpy as np
import torch
import torch.distributions
from torch.distributions import Categorical, MultivariateNormal, TransformedDistribution
from torch.distributions.transforms import AffineTransform, TanhTransform

from rllib.util.distributions import Delta


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


def mellow_max(values, omega=1.0):
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


def tensor_to_distribution(args, **kwargs):
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
        if kwargs.get("add_noise", False):
            noise_clip = kwargs.get("noise_clip", np.inf)
            policy_noise = kwargs.get("policy_noise", 1)
            action_scale = kwargs.get("action_scale", 1)
            try:
                policy_noise = policy_noise()
            except TypeError:
                pass
            noise_scale = policy_noise * action_scale
            mean = args[0] + (torch.randn_like(args[0]) * noise_scale).clamp(
                -noise_clip, noise_clip
            )
        else:
            mean = args[0]
        return Delta(mean, event_dim=min(1, mean.dim()))
    else:
        if kwargs.get("tanh", False):
            d = TransformedDistribution(
                MultivariateNormal(args[0], scale_tril=args[1]),
                [
                    AffineTransform(loc=0, scale=1 / kwargs.get("action_scale", 1)),
                    TanhTransform(),
                    AffineTransform(loc=0, scale=kwargs.get("action_scale", 1)),
                ],
            )
        elif kwargs.get("normalized", False):
            scale = kwargs.get("action_scale", 1)
            d = MultivariateNormal(args[0] / scale, scale_tril=args[1] / scale)
        else:
            d = MultivariateNormal(args[0], scale_tril=args[1])
        return d


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
        KL divergence that corresponds to a shift in the mean components while keeping
        the scale fixed.
    kl_var: torch.Tensor
        KL divergence that corresponds to a shift in the scale components while keeping
        the location fixed.
    """
    kl_mean = torch.distributions.kl_divergence(
        p=MultivariateNormal(p.loc, scale_tril=q.scale_tril), q=q
    ).mean()
    kl_var = torch.distributions.kl_divergence(
        p=MultivariateNormal(q.loc, scale_tril=p.scale_tril), q=q
    ).mean()

    return kl_mean, kl_var


def sample_mean_and_cov(sample, diag=False):
    """Compute mean and covariance of a sample of vectors.

    Parameters
    ----------
    sample: Tensor
        Tensor of dimensions [batch x N x num_samples].
    diag: bool, optional.
        Flag to indicate if the computation has to assume independent or correlated
        variables.

    Returns
    -------
    mean: Tensor
        Tensor of dimension [batch x N].
    covariance: Tensor
        Tensor of dimension [batch x N x N].

    """
    num_samples = sample.shape[-1]
    mean = torch.mean(sample, dim=-1, keepdim=True)

    if diag:
        covariance = torch.diag_embed(sample.var(-1))
    else:
        sigma = (mean - sample) @ (mean - sample).transpose(-2, -1)
        sigma += 1e-6 * torch.eye(sigma.shape[-1])  # Add some jitter.
        covariance = sigma / num_samples
    mean = mean.squeeze(-1)

    return mean, covariance


def safe_cholesky(covariance_matrix, jitter=1e-6):
    """Perform a safe cholesky decomposition of the covariance matrix.

    If cholesky decomposition raises Runtime error, it adds jitter to the covariance
    matrix.

    Parameters
    ----------
    covariance_matrix: torch.Tensor.
        Tensor with dimensions batch x dim x dim.
    jitter: float, optional.
        Jitter to add to the covariance matrix.
    """
    try:
        return torch.cholesky(covariance_matrix)
    except RuntimeError:
        dim = covariance_matrix.shape[-1]
        if jitter > 1:
            # When jitter is too big, then there is some numerical issue and this avoids
            # stack overflow.
            warnings.warn("Jitter too big. Maybe some numerical issue somewhere.")
            return torch.eye(dim)
        return safe_cholesky(
            covariance_matrix + jitter * torch.eye(dim), jitter=10 * jitter
        )


def moving_average_filter(x, y, horizon):
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
    x_smooth = x[horizon // 2 : -horizon // 2 + 1]
    y_smooth = np.convolve(y, smoothing_weights, "valid")
    return x_smooth, y_smooth


class RewardTransformer(object):
    r"""Reward Transformer.

    Compute a reward by appling:
    ..math:: out = (scale * (reward - offset)).clamp(min, max)
    """

    def __init__(self, offset=0, low=-np.inf, high=np.inf, scale=1.0):
        self.offset = offset
        self.low = low
        self.high = high
        self.scale = scale

    def __call__(self, reward):
        """Call transformation function."""
        if isinstance(reward, torch.Tensor):
            return (self.scale * (reward - self.offset)).clamp(self.low, self.high)
        else:
            return np.clip(self.scale * (reward - self.offset), self.low, self.high)
