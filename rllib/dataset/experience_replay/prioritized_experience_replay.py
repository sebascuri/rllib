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
                 batch_size=1, max_priority=10., transformations=None):
        super().__init__(max_len, batch_size, transformations)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.beta_increment = beta_inc
        self.max_priority = max_priority
        self.priorities = np.empty((self.max_len,), dtype=np.float)

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

    def update(self, indexes, td_error):
        """Update experience replay sampling distribution with set of weights."""
        priority = self._get_priority(td_error)
        self.priorities[indexes] = priority

    def _get_priority(self, td_error):
        return (np.abs(td_error) + self.epsilon) ** self.alpha

    def get_batch(self, batch_size=None):
        """Get a batch of data."""
        batch_size = batch_size if batch_size is not None else self.batch_size

        self.beta = np.min([1., self.beta + self.beta_increment])
        num = len(self)
        probs = self.priorities[:num]
        probs = probs / np.sum(probs)
        indices = np.random.choice(num, batch_size, p=probs)

        probs = probs[indices]
        weights = np.power(probs * num, -self.beta)
        weights /= weights.max()

        return default_collate([self[i] for i in indices]), indices, weights
