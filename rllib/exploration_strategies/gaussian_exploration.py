"""Additive Gaussian Noise Exploration Strategy."""

from .abstract_exploration_strategy import AbstractExplorationStrategy
import numpy as np


__all__ = ['GaussianExploration']


class GaussianExploration(AbstractExplorationStrategy):
    """Implementation of Additive Gaussian Noise Exploration strategy."""

    def __call__(self, action_distribution, steps=None):
        """See `AbstractExplorationStrategy.__call__'."""
        std_dev = self.param(steps)
        mean = action_distribution.mean.detach().numpy()
        noise = std_dev * np.random.randn()
        return mean + noise
