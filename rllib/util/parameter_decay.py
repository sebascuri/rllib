"""Implementation of a Parameter decay class."""

from abc import ABCMeta

import torch.jit
import torch.nn as nn

from rllib.util.neural_networks.utilities import inverse_softplus


class ParameterDecay(nn.Module, metaclass=ABCMeta):
    """Abstract class that implements the decay of a parameter."""

    def __init__(self, start, end=None, decay=None):
        super().__init__()
        if not isinstance(start, torch.Tensor):
            start = torch.tensor(start)
        self.start = nn.Parameter(start, requires_grad=False)

        if end is None:
            end = start
        if not isinstance(end, torch.Tensor):
            end = torch.tensor(end)
        self.end = nn.Parameter(end, requires_grad=False)

        if decay is None:
            decay = 1.0
        if not isinstance(decay, torch.Tensor):
            decay = torch.tensor(decay)
        self.decay = nn.Parameter(decay, requires_grad=False)

        self.step = 0

    @torch.jit.export
    def update(self):
        """Update parameter."""
        self.step += 1


class Constant(ParameterDecay):
    """Constant parameter."""

    def forward(self):
        """See `ParameterDecay.__call__'."""
        return self.start


class Learnable(ParameterDecay):
    """Learnable parameter."""

    positive: bool

    def __init__(self, val, positive: bool = False):
        self.positive = positive
        if self.positive:
            val = inverse_softplus(val).item()
        super().__init__(val)
        self.start.requires_grad = True
        self.positive = positive

    def forward(self):
        """See `ParameterDecay.__call__'."""
        if self.positive:
            return torch.nn.functional.softplus(self.start) + 1e-4
        else:
            return self.start


class ExponentialDecay(ParameterDecay):
    """Exponential decay of parameter."""

    def forward(self):
        """See `ParameterDecay.__call__'."""
        decay = torch.exp(-torch.tensor(1.0 * self.step) / self.decay)
        return self.end + (self.start - self.end) * decay


class PolynomialDecay(ParameterDecay):
    """Polynomial Decay of a parameter.

    It returns the minimum between start and end / step ** decay.
    """

    def forward(self):
        """See `ParameterDecay.__call__'."""
        return min(self.start, self.end / torch.tensor(self.step + 1.0) ** self.decay)


class LinearDecay(ParameterDecay):
    """Linear decay of parameter."""

    def forward(self):
        """See `ParameterDecay.__call__'."""
        return max(self.end, self.start - self.decay * self.step)


class LinearGrowth(ParameterDecay):
    """Linear decay of parameter."""

    def forward(self):
        """See `ParameterDecay.__call__'."""
        return min(self.end, self.start + self.decay * self.step)


class OUNoise(ParameterDecay):
    """Ornstein-Uhlenbeck Noise process.

    Parameters
    ----------
    mean: Tensor
        Mean of OU process.
    std_deviation: Tensor
        Standard Deviation of OU Process.
    theta: float
        Parameter of OU Process.
    dt: float
        Time discretization.
    dim: Tuple
        Dimensions of noise.

    References
    ----------
    https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
    """

    def __init__(self, mean=0, std_deviation=0.2, theta=0.15, dt=1e-2, dim=(1,)):
        if not isinstance(mean, torch.Tensor):
            mean = mean * torch.ones(dim)
        self.mean = mean

        if not isinstance(std_deviation, torch.Tensor):
            std_deviation = std_deviation * torch.ones(dim)
        self.std_dev = std_deviation

        self.theta = theta
        self.dt = dt
        super().__init__(start=torch.zeros_like(mean))

    def forward(self):
        """Compute Ornstein-Uhlenbeck sample."""
        x_prev = self.start.data

        x = (
            x_prev
            + self.theta * (self.mean - x_prev) * self.dt
            + self.std_dev
            * torch.sqrt(torch.tensor(self.dt))
            * torch.randn(self.mean.shape)
        )
        self.start.data = x
        return x
