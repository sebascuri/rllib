"""Additive Gaussian Noise Exploration Strategy."""

from .abstract_exploration_strategy import AbstractExplorationStrategy
import numpy as np


class GaussianNoise(AbstractExplorationStrategy):
    """Implementation of Additive Gaussian Noise Exploration strategy."""

    def __call__(self, state=None):
        """See `AbstractExplorationStrategy.__call__'."""
        noise = self.param() * np.random.randn(self.dimension)
        return noise
