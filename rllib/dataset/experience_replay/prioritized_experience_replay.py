"""Implementation of a Prioritized Experience Replay Buffer."""

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from .experience_replay import ExperienceReplay
from rllib.util.parameter_decay import Constant


class PrioritizedExperienceReplay(ExperienceReplay):
    """Implementation of Prioritized Experience Replay Algorithm.

    References
    ----------
    Schaul, Tom, et al. "PRIORITIZED EXPERIENCE REPLAY." ICLR 2016.

    """

    def __init__(self, max_len, alpha=0.6, beta=0.4, epsilon=0.01, max_priority=10.,
                 transformations=None):
        super().__init__(max_len, transformations)
        if isinstance(alpha, float):
            alpha = Constant(alpha)
        self.alpha = alpha

        if isinstance(beta, float):
            beta = Constant(beta)
        self.beta = beta

        self.epsilon = torch.tensor(epsilon)
        self.max_priority = max_priority
        self.priorities = torch.zeros(self.max_len)
        self.weights = torch.zeros(self.max_len)

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
        probs = self.priorities[:num] / torch.sum(self.priorities[:num])

        weights = torch.pow(probs * num, -self.beta())
        self.weights[:num] = weights

    def update(self, indexes, td_error):
        """Update experience replay sampling distribution with set of weights."""
        self.priorities[indexes] = self._get_priority(td_error)
        self.alpha.update()
        self.beta.update()
        self._update_weights()

    def _get_priority(self, td_error):
        return (torch.abs(td_error) + self.epsilon) ** self.alpha()

    def get_batch(self, batch_size):
        """Get a batch of data."""
        probs = self.priorities[:len(self)].numpy()
        indices = np.random.choice(len(self), batch_size, p=probs / np.sum(probs))
        return default_collate([self[i] for i in indices])
