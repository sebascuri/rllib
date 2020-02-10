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

    @abstractmethod
    def __call__(self, steps=None):
        """Call parameter value at a given number of steps."""
        raise NotImplementedError


class ExponentialDecay(ParameterDecay):
    """Exponential decay of parameter."""

    def __call__(self, steps=None):
        """See `ParameterDecay.__call__'."""
        if steps is None:
            return self.start
        else:
            decay = (self.start - self.end) * np.exp(-steps / self.decay)
            return self.end + decay


class LinearDecay(ParameterDecay):
    """Linear decay of parameter."""

    def __call__(self, steps=None):
        """See `ParameterDecay.__call__'."""
        if steps is not None:
            self.start = max(self.end, self.start - self.decay)

        return self.start
