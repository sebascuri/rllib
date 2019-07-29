from .abstract_exploration_strategy import AbstractExplorationStrategy
import numpy as np
from torch.distributions import Categorical


def _mode(action_distribution):
    if type(action_distribution) is Categorical:  # Categorical
        return action_distribution.logits.argmax().item()
    else:  # MultivariateNormal
        return action_distribution.loc.numpy()


def _random_sample(action_distribution, scale=1.0):
    if type(action_distribution) is Categorical:  # Categorical
        return np.random.choice(len(action_distribution.logits))
    else:
        return action_distribution.sample().numpy() * scale


class EpsGreedy(AbstractExplorationStrategy):
    def __init__(self, eps_start, eps_end=None, eps_decay=None, scale=1):
        self._eps_start = eps_start

        if eps_end is None:
            eps_end = eps_start
        self._eps_end = eps_end

        if eps_decay is None:
            eps_decay = 1
        self._eps_decay = eps_decay

        self._scale = scale

    def __str__(self):
        return "Epsilon-Greedy"

    def __call__(self, action_distribution, steps=None):
        if np.random.random() > self.epsilon(steps):
            return _mode(action_distribution)
        else:
            return _random_sample(action_distribution, self._scale)

    def epsilon(self, steps=None):
        if steps is None:
            return self._eps_start
        else:
            decay = (self._eps_start - self._eps_end) * np.exp(-steps / self._eps_decay)
            return self._eps_end + decay
