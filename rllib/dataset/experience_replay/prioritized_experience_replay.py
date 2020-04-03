"""Implementation of a Prioritized Experience Replay Buffer."""

import numpy as np
from torch.utils.data._utils.collate import default_collate

from .experience_replay import ExperienceReplay


class PrioritizedExperienceReplay(ExperienceReplay):
    """Implementation of Prioritized Experience Replay Algorithm.

    References
    ----------
    Schaul, Tom, et al. "PRIORITIZED EXPERIENCE REPLAY." ICLR 2016.

    """

    def __init__(self, max_len, alpha=0.6, beta=0.4, epsilon=0.01, beta_inc=0.001,
                 max_priority=10., transformations=None):
        super().__init__(max_len, transformations)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.beta_increment = beta_inc
        self.max_priority = max_priority
        self.priorities = np.zeros((self.max_len,), dtype=np.float)
        self.weights = np.zeros((self.max_len,), dtype=np.float)

    def append(self, observation):
        """Append new observation to the dataset.

        Parameters
        ----------
        observation: Observation

        Raises
        ------
        TypeError
            If the new observation is not of type Observation.
        """
        self.priorities[self._ptr] = self.max_priority
        super().append(observation)
        self._update_weights()

    def _update_weights(self):
        """Update priorities and weights."""
        num = len(self)
        probs = self.priorities[:num] / np.sum(self.priorities[:num])

        weights = np.power(probs * num, -self.beta)
        self.weights[:num] = weights / weights.max()

    def update(self, indexes, td_error):
        """Update experience replay sampling distribution with set of weights."""
        self.priorities[indexes] = self._get_priority(td_error)
        self.beta = np.min([1., self.beta + self.beta_increment])
        self._update_weights()

    def _get_priority(self, td_error):
        return (np.abs(td_error) + self.epsilon) ** self.alpha

    def get_batch(self, batch_size):
        """Get a batch of data."""
        probs = self.priorities[:len(self)]
        indices = np.random.choice(len(self), batch_size, p=probs/np.sum(probs))
        return default_collate([self[i] for i in indices])
