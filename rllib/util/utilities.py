"""Utilities for the rllib library."""
import pickle
import time
import warnings

import numpy as np
import torch
import torch.distributions
from torch.distributions import Categorical, MultivariateNormal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

from rllib.util.distributions import Delta


def get_backend(array):
    """Get backend of the array."""
    if isinstance(array, torch.Tensor):
        return torch
    elif isinstance(array, np.ndarray):
        return np
    else:
        raise TypeError


def set_random_seed(seed):
    """Set global random seed."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_random_state(directory):
    """Save the simulation random state in a directory."""
    with open(f"{directory}/random_state.pkl", "wb") as f:
        pickle.dump({"numpy": np.random.get_state(), "torch": torch.get_rng_state()}, f)


def load_random_state(directory):
    """Load the simulation random state from a directory."""
    with open(f"{directory}/random_state.pkl", "rb") as f:
        random_states = pickle.load(f)

    if "numpy" in random_states:
        np.random.set_state(random_states["numpy"])

    if "torch" in random_states:
        torch.set_rng_state(random_states["torch"])


def integrate(function, distribution, out_dim=None, num_samples=15):
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
        Distribution to integrate the function w.r.t.
    out_dim: int, optional.
    num_samples: int.
        Number of samples in MC integration.

    Returns
    -------
    integral value.
    """
    batch_size = distribution.batch_shape
    if out_dim is None:
        ans = torch.zeros(batch_size)
    else:
        ans = torch.zeros(batch_size + (out_dim,))

    if distribution.has_enumerate_support:
        for action in distribution.enumerate_support():
            prob = distribution.probs.gather(-1, action.unsqueeze(-1))
            f_val = function(action)
            if out_dim is None:
                prob = prob.squeeze(-1)
            ans += prob.detach() * f_val

    else:
        for _ in range(num_samples):
            if distribution.has_rsample:
                action = distribution.rsample()
            else:
                action = distribution.sample()
            f_val = function(action)
            if f_val.ndim > ans.ndim:
                f_val = f_val.squeeze(-1)
            ans += f_val / num_samples
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
            try:
                policy_noise = policy_noise()
            except TypeError:
                pass
            mean = args[0] + (torch.randn_like(args[0]) * policy_noise).clamp(
                -noise_clip, noise_clip
            )
        else:
            mean = args[0]
        return Delta(v=mean, event_dim=min(1, mean.dim()))
    else:
        if kwargs.get("tanh", False):
            d = TransformedDistribution(
                MultivariateNormal(args[0], scale_tril=args[1]), [TanhTransform()]
            )
        else:
            d = MultivariateNormal(args[0], scale_tril=args[1])
        return d


def separated_kl(p, q, log_p=torch.tensor(0.0), log_q=torch.tensor(0.0)):
    """Compute the mean and variance components of the average KL divergence.

    Separately computes the mean and variance contributions to the KL divergence
    KL(p || q).

    Parameters
    ----------
    p: torch.distributions.MultivariateNormal
    q: torch.distributions.MultivariateNormal
    log_p: torch.Tensor.
    log_q: torch.Tensor.

    Returns
    -------
    kl_mean: torch.Tensor
        KL divergence that corresponds to a shift in the mean components while keeping
        the scale fixed.
    kl_var: torch.Tensor
        KL divergence that corresponds to a shift in the scale components while keeping
        the location fixed.
    """
    assert isinstance(p, type(q))
    if isinstance(p, torch.distributions.MultivariateNormal) and isinstance(
        q, torch.distributions.MultivariateNormal
    ):
        kl_mean = torch.distributions.kl_divergence(
            p=MultivariateNormal(p.loc, scale_tril=q.scale_tril), q=q
        )
        kl_var = torch.distributions.kl_divergence(
            p=MultivariateNormal(q.loc, scale_tril=p.scale_tril), q=q
        )
    elif isinstance(p, Delta):
        kl_mean = 0.5 * (p.mean - q.mean).square().mean(-1)
        kl_var = torch.zeros_like(kl_mean)
    else:
        try:
            kl_mean = torch.distributions.kl_divergence(p=p, q=q)
        except NotImplementedError:
            kl_mean = log_p - log_q  # Approximate the KL with samples.
        kl_var = torch.zeros_like(kl_mean)

    return kl_mean, kl_var


def off_policy_weight(eval_log_p, behavior_log_p, full_trajectory=False, clamp_max=5.0):
    """Compute off-policy weight.

    Parameters
    ----------
    eval_log_p: torch.tensor.
        Evaluation log probabilities.
    behavior_log_p: torch.tensor.
        Behavior log probabilities.
    full_trajectory: bool, optional (default=False).
        Flag that indicates whether the off-policy weight is for a single step or for
        the full trajectory.
    clamp_max: float.
        Value to clamp max.

    Returns
    -------
    weight: torch.Tensor.
        Importance sample weights of the trajectory.
    """
    weight = torch.exp(eval_log_p - behavior_log_p)
    if full_trajectory:
        weight = torch.cumprod(weight, dim=-1)

    return weight.clamp_max(clamp_max)


def get_entropy_and_log_p(pi, action, action_scale):
    """Get the entropy and the log-probability of a policy and an action."""
    if isinstance(action_scale, torch.Tensor):
        action_scale = action_scale.clone()
        action_scale[action_scale == 0] = 1.0
    log_p = pi.log_prob(action / action_scale)

    try:
        entropy = pi.entropy()
    except NotImplementedError:
        entropy = -log_p  # Approximate with the sampled action (biased).

    return entropy, log_p


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


class MovingAverage(object):
    """Calculate moving average."""

    def __init__(self):
        self._count = 0
        self._total_value = 0.0

    def update(self, value):
        """Update moving average."""
        self._count += 1
        self._total_value += value

    @property
    def value(self):
        """Get current value."""
        if self._count > 0:
            return self._total_value / self._count
        else:
            return float("inf")


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


class TimeIt(object):
    """Class to time a piece of code."""

    def __init__(self, name=""):
        self.start = 0
        self.name = name

    def __enter__(self):
        """Start counting the time."""
        self.start = time.time()

    def __exit__(self, *args):
        """Print the elapsed time."""
        print(f"Elapsed time doing {self.name}: {time.time() - self.start}")
