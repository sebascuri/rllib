"""Implementation of a Parameter decay class."""

from abc import ABCMeta

import torch.jit
import torch.nn as nn


class ParameterDecay(nn.Module, metaclass=ABCMeta):
    """Abstract class that implements the decay of a parameter."""

    def __init__(self, start, end=None, decay=None):
        super().__init__()
        self.start = nn.Parameter(torch.tensor(start), requires_grad=False)

        if end is None:
            end = start
        self.end = nn.Parameter(torch.tensor(end), requires_grad=False)

        if decay is None:
            decay = 1.
        self.decay = nn.Parameter(torch.tensor(decay), requires_grad=False)

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

    def __init__(self, val):
        super().__init__(val)
        self.start.requires_grad = True

    def forward(self):
        """See `ParameterDecay.__call__'."""
        return self.start


class ExponentialDecay(ParameterDecay):
    """Exponential decay of parameter."""

    def forward(self):
        """See `ParameterDecay.__call__'."""
        decay = torch.exp(-torch.tensor(1. * self.step) / self.decay)
        return self.end + (self.start - self.end) * decay


class PolynomialDecay(ParameterDecay):
    """Polynomial Decay of a parameter.

    It returns the minimum between start and end / step ** decay.
    """

    def forward(self):
        """See `ParameterDecay.__call__'."""
        return min(self.start, self.end / torch.tensor(self.step + 1.) ** self.decay)


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
