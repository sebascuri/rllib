"""Implementation of an Experience Replay Buffer."""


import numpy as np
from . import Observation
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from typing import List


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

    """

    def __init__(self, max_len, batch_size=1, transformations: list = None):
        super().__init__()
        self._max_len = max_len
        self._memory = np.empty((self._max_len,), dtype=Observation)
        self._sampling_idx = []  # type: List[int]
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
        # assert idx < len(self)
        idx = self._sampling_idx[idx]
        """This is done for shuffling. If memory was directly shuffled, then ordering
        would be lost.
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
        if not self.is_full:
            self._sampling_idx.append(self._ptr)
        self._memory[self._ptr] = observation
        self._ptr = (self._ptr + 1) % self._max_len

        for transformation in self._transformations:
            transformation.update(observation)

    def shuffle(self):
        """Shuffle the dataset."""
        np.random.shuffle(self._sampling_idx)

    def get_batch(self):
        """Get a batch of data."""
        indices = np.random.choice(len(self), self.batch_size)
        return default_collate([self[i] for i in indices])

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
