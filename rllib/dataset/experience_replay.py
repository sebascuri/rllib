"""Implementation of an Experience Replay Buffer."""

import numpy as np
from . import Observation
from torch.utils import data
from torch.utils.data._utils.collate import default_collate

__all__ = ['ExperienceReplay', 'PrioritizedExperienceReplay',
           'LinfSampler', 'L1Sampler']


class ExperienceReplay(data.Dataset):
    """An Experience Replay Buffer dataset.

    The Experience Replay algorithm stores transitions and access them IID. It has a
    size, and it erases the older samples, once the buffer is full, like on a queue.

    Parameters
    ----------
    max_len: int.
        buffer size of experience replay algorithm.
    batch_size: int.
        batch size to sample.
    transformations: list of transforms.AbstractTransform, optional.
        A sequence of transformations to apply to the dataset, each of which is a
        callable that takes an observation as input and returns a modified observation.
        If they have an `update` method it will be called whenever a new trajectory
        is added to the dataset.

    Methods
    -------
    append(observation):
        append an observation to the dataset.
    shuffle():
        shuffle the dataset.
    is_full: bool
        check if buffer is full.
    has_batch: bool
        check if buffer has at least a batch.
    update(observation, indexes, priority):
        update experience replay sampling distribution with priority.

    """

    def __init__(self, max_len, batch_size=1, transformations: list = None):
        super().__init__()
        self._max_len = max_len
        self._memory = np.empty((self._max_len,), dtype=Observation)
        self._ptr = 0
        self._transformations = transformations or list()
        self.batch_size = batch_size

    def __getitem__(self, idx):
        """Return any desired observation.

        Parameters
        ----------
        idx: int

        Returns
        -------
        observation: Observation

        """
        observation = self._memory[idx]
        for transform in self._transformations:
            observation = transform(observation)
        return observation

    def __len__(self):
        """Return the current size of the buffer."""
        if self.is_full:
            return self._max_len
        else:
            return self._ptr

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
        if not type(observation) == Observation:
            raise TypeError("""
            input has to be of type Observation, and it was found of type {}
            """.format(type(observation)))

        self._memory[self._ptr] = observation
        self._ptr = (self._ptr + 1) % self._max_len

        for transformation in self._transformations:
            transformation.update(observation)

    def get_batch(self, batch_size=None):
        """Get a batch of data."""
        batch_size = batch_size if batch_size is not None else self.batch_size
        indices = np.random.choice(len(self), batch_size)
        weights = np.ones(batch_size)
        return default_collate([self[i] for i in indices]), indices, weights

    @property
    def is_full(self):
        """Flag that checks if memory in buffer is full.

        Returns
        -------
        bool
        """
        return self._memory[-1] is not None  # check if the last element is not empty.

    @property
    def has_batch(self):
        """Return true if there are more examples than the batch size."""
        return len(self) >= self.batch_size

    def update(self, indexes, td_error):
        """Update experience replay sampling distribution with set of weights."""
        pass


class PrioritizedExperienceReplay(ExperienceReplay):
    """Implementation of Prioritized Experience Replay Algorithm.

    References
    ----------
    Schaul, Tom, et al. "PRIORITIZED EXPERIENCE REPLAY." ICLR 2016.

    """

    def __init__(self, max_len, alpha=0.6, beta=0.4, epsilon=0.01, beta_inc=0.001,
                 batch_size=1, max_priority=10., transformations: list = None):
        super().__init__(max_len, batch_size, transformations)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.beta_increment = beta_inc
        self.max_priority = max_priority
        self._priorities = np.empty((self._max_len,), dtype=np.float)

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
        self._priorities[self._ptr] = self.max_priority
        super().append(observation)

    def update(self, indexes, td_error):
        """Update experience replay sampling distribution with set of weights."""
        priority = self._get_priority(td_error)
        self._priorities[indexes] = priority

    def _get_priority(self, td_error):
        return (np.abs(td_error) + self.epsilon) ** self.alpha

    def get_batch(self, batch_size=None):
        """Get a batch of data."""
        batch_size = batch_size if batch_size is not None else self.batch_size

        self.beta = np.min([1., self.beta + self.beta_increment])
        num = len(self)
        probs = self._priorities[:num]
        probs = probs / np.sum(probs)
        indices = np.random.choice(num, batch_size, p=probs)

        probs = probs[indices]
        weights = np.power(probs * num, -self.beta)
        weights /= weights.max()

        return default_collate([self[i] for i in indices]), indices, weights


class LinfSampler(ExperienceReplay):
    """Sampler for L-infinity Algorithm."""

    def __init__(self, max_len, eta=0.1, beta=0.1, batch_size=1, max_priority=1.,
                 transformations: list = None):
        super().__init__(max_len, batch_size, transformations)
        self.eta = eta
        self.beta = beta
        self.max_priority = max_priority
        self._priorities = np.empty((self._max_len,), dtype=np.float)

    def append(self, observation):
        """Append new observations."""
        self._priorities[self._ptr] = self.max_priority
        super().append(observation)

    def update(self, indexes, td_error):
        """Update experience replay sampling distribution with set of weights."""
        # Implement this way or in the primal space?
        self._priorities[indexes] += self.eta * td_error / self.probabilities(indexes)

    def probabilities(self, indexes=None, sign=1):
        """Get probabilities of a given set of indexes."""
        num = len(self)
        if indexes is None:
            indexes = np.arange(num)
        probs = np.exp(sign * self._priorities[:num])
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


class L1Sampler(LinfSampler):
    """Sampler for L1 Algorithm."""

    def get_batch(self, batch_size=None):
        """Get a batch of data."""
        batch_size = batch_size if batch_size is not None else self.batch_size
        num = len(self)
        pprobs = self.probabilities(sign=1)
        nprobs = self.probabilities(sign=-1)
        probs = 1 / 2 * (pprobs + nprobs)
        indices = np.random.choice(num, batch_size, p=probs)

        return default_collate([self[i] for i in indices]), indices, 1 / pprobs[indices]
