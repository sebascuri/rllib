"""Implementation of an EXP3 Experience Replay Buffer."""

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from .experience_replay import ExperienceReplay
from rllib.util.parameter_decay import Constant


class EXP3ExperienceReplay(ExperienceReplay):
    """Sampler for EXP3-Sampler."""

    def __init__(self, max_len, eta=0.1, gamma=0.1, max_priority=1.,
                 transformations=None):
        super().__init__(max_len, transformations)
        if isinstance(eta, float):
            eta = Constant(eta)
        self.eta = eta

        if isinstance(gamma, float):
            gamma = Constant(gamma)
        self.gamma = gamma

        self.max_priority = max_priority
        self.priorities = torch.zeros(self.max_len)
        self.weights = torch.zeros(self.max_len)

    def append(self, observation):
        """Append new observations."""
        self.priorities[self._ptr] = self.max_priority
        super().append(observation)
        self._update_weights()

    def update(self, indexes, td_error):
        """Update experience replay sampling distribution with set of weights."""
        idx, inverse_idx, counts = torch.unique(indexes, return_counts=True,
                                                return_inverse=True)

        td = torch.abs(td_error)
        inv_prob = self.weights[indexes]
        self.priorities[indexes] += self.eta() * td * inv_prob * counts[inverse_idx]

        self.max_priority = max(self.max_priority,
                                torch.max(self.priorities[idx]).item())
        self.eta.update()
        self.gamma.update()
        self._update_weights()

    def _update_weights(self):
        """Update priorities and weights."""
        num = len(self)
        probs = torch.exp(self.priorities[:num] - self.max_priority)
        probs = (1 - self.gamma()) * probs / torch.sum(probs) + self.gamma() / num
        weights = 1. / (num * probs)
        self.weights[:num] = weights

    def get_batch(self, batch_size):
        """Get a batch of data."""
        probs = 1. / self.weights[:len(self)].numpy()
        # assert np.all(probs + 1e-3 >= self.gamma().data.item())
        indices = np.random.choice(len(self), batch_size, p=probs / np.sum(probs))
        return default_collate([self[i] for i in indices])
