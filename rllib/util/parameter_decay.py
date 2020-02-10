"""Implementation of a Parameter decay class."""

import numpy as np
from abc import ABC, abstractmethod


class ParameterDecay(ABC):
    """Abstract class that implements the decay of a parameter."""

    def __init__(self, start, end=None, decay=None):
        self.start = start

        if end is None:
            end = start
        self.end = end

        if decay is None:
            decay = 1
        self.decay = decay

        self.step = 0

    @abstractmethod
    def __call__(self):
        """Call parameter value at a given number of steps."""
        raise NotImplementedError


class ExponentialDecay(ParameterDecay):
    """Exponential decay of parameter."""

    def __call__(self):
        """See `ParameterDecay.__call__'."""
        decay = (self.start - self.end) * np.exp(-self.step / self.decay)
        self.step += 1
        return self.end + decay


class LinearDecay(ParameterDecay):
    """Linear decay of parameter."""

    def __call__(self):
        """See `ParameterDecay.__call__'."""
        self.step += 1
        self.start = max(self.end, self.start - self.decay)

        return self.start
