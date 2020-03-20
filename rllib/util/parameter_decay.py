"""Implementation of a Parameter decay class."""

from abc import ABCMeta

import torch.jit
import torch.nn as nn


class ParameterDecay(nn.Module, metaclass=ABCMeta):
    """Abstract class that implements the decay of a parameter."""

    def __init__(self, start, end=None, decay=None):
        super().__init__()
        self.start = start

        if end is None:
            end = start
        self.end = end

        if decay is None:
            decay = 1
        self.decay = decay

        self.step = 0

    @torch.jit.export
    def update(self):
        """Update parameter."""
        pass


class Constant(ParameterDecay):
    """Constant parameter."""

    def forward(self):
        """See `ParameterDecay.__call__'."""
        return self.start

    def update(self):
        """Update parameter."""
        pass


class ExponentialDecay(ParameterDecay):
    """Exponential decay of parameter."""

    def forward(self):
        """See `ParameterDecay.__call__'."""
        decay = torch.exp(-torch.tensor(self.step / self.decay))
        return self.end + (self.start - self.end) * decay

    def update(self):
        """Update parameter."""
        self.step += 1


class LinearDecay(ParameterDecay):
    """Linear decay of parameter."""

    def forward(self):
        """See `ParameterDecay.__call__'."""
        return self.start

    def update(self):
        """Update parameter."""
        self.step += 1
        self.start = max(self.end, self.start - self.decay)
