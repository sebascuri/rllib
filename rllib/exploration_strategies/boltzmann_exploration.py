from .abstract_exploration_strategy import AbstractExplorationStrategy
from torch.distributions import Categorical, MultivariateNormal
import numpy as np


class BoltzmannExploration(AbstractExplorationStrategy):
    def __init__(self, t_start, t_end=None, t_decay=None):
        self._t_start = t_start

        if t_end is None:
            t_end = t_start
        self._t_end = t_end

        if t_decay is None:
            t_decay = 1
        self._t_decay = t_decay

    def __str__(self):
        return "Boltzmann Exploration"

    def __call__(self, action_distribution, steps=None):
        t = self.temperature(steps)
        if type(action_distribution) is Categorical:
            d = Categorical(logits=action_distribution.logits / t)
        else:
            d = MultivariateNormal(
                loc=action_distribution.loc,
                covariance_matrix=action_distribution.covariance_matrix * t)

        return d.sample().numpy()

    def temperature(self, steps=None):
        if steps is None:
            return self._t_start
        else:
            decay = (self._t_start - self._t_end) * np.exp(-steps / self._t_decay)
            return self._t_end + decay
