"""Implementation of an Experience Replay Buffer."""

import numpy as np
# import torch
from torch.utils import data
from torch.utils.data._utils.collate import default_collate

from rllib.dataset.datatypes import Observation
from rllib.dataset.utilities import stack_list_of_tuples


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

    def __init__(self, max_len, batch_size=1, transformations=None):
        super().__init__()
        self.max_len = max_len
        self.memory = np.empty((self.max_len,), dtype=Observation)
        self._ptr = 0
        self.transformations = transformations or list()
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
        observation = self.memory[idx]
        for transform in self.transformations:
            observation = transform(observation)
        return observation

    def reset(self):
        """Reset memory to empty."""
        self.memory = np.empty((self.max_len,), dtype=Observation)
        self._ptr = 0

    def __len__(self):
        """Return the current size of the buffer."""
        if self.is_full:
            return self.max_len
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
            raise TypeError(
                f"input has to be of type Observation, and it was {type(observation)}")

        self.memory[self._ptr] = observation
        self._ptr = (self._ptr + 1) % self.max_len

        for transformation in self.transformations:
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
        return self.memory[-1] is not None  # check if the last element is not empty.

    @property
    def all_data(self):
        """Get all the data."""
        data = stack_list_of_tuples(self.memory[:self._ptr])

        for transformation in self.transformations:
            data = transformation(data)
        return data

    @property
    def has_batch(self):
        """Return true if there are more examples than the batch size."""
        return len(self) >= self.batch_size

    def update(self, indexes, td_error):
        """Update experience replay sampling distribution with set of weights."""
        pass
