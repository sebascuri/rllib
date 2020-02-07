"""Implementation of an EXP3 Experience Replay Buffer."""

import numpy as np
from .experience_replay import ExperienceReplay
from torch.utils.data._utils.collate import default_collate


class EXP3Sampler(ExperienceReplay):
    """Sampler for EXP3-Sampler Algorithm."""

    def __init__(self, max_len, eta=0.1, beta=0.1, batch_size=1, max_priority=1.,
                 transformations=None):
        super().__init__(max_len, batch_size, transformations)
        self.eta = eta
        self.beta = beta
        self.max_priority = max_priority
        self.priorities = np.empty((self.max_len,), dtype=np.float)

    def append(self, observation):
        """Append new observations."""
        self.priorities[self._ptr] = self.max_priority
        super().append(observation)

    def update(self, indexes, td_error):
        """Update experience replay sampling distribution with set of weights."""
        # Implement this way or in the primal space?
        self.priorities[indexes] += self.eta * td_error / self.probabilities(indexes)

    def probabilities(self, indexes=None, sign=1):
        """Get probabilities of a given set of indexes."""
        num = len(self)
        if indexes is None:
            indexes = np.arange(num)
        probs = np.exp(sign * self.priorities[:num])
        probs = probs / np.sum(probs)
        probs = (1 - self.beta) * probs + self.beta / num
        return probs[indexes]

    def get_batch(self, batch_size=None):
        """Get a batch of data."""
        batch_size = batch_size if batch_size is not None else self.batch_size
        num = len(self)
        probs = self.probabilities()
        indices = np.random.choice(num, batch_size, p=probs)

        return default_collate([self[i] for i in indices]), indices, 1 / probs[indices]
