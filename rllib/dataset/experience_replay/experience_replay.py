"""Implementation of an Experience Replay Buffer."""

import numpy as np
import torch
from torch.utils import data
from torch.utils.data._utils.collate import default_collate

from rllib.dataset.datatypes import Observation
from rllib.dataset.utilities import stack_list_of_tuples, concatenate_observations


class ExperienceReplay(data.Dataset):
    """An Experience Replay Buffer dataset.

    The Experience Replay algorithm stores transitions and access them IID. It has a
    size, and it erases the older samples, once the buffer is full, like on a queue.

    Parameters
    ----------
    max_len: int.
        buffer size of experience replay algorithm.
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

    def __init__(self, max_len, transformations=None, num_steps=1):
        super().__init__()
        self.max_len = max_len
        self.memory = np.empty((self.max_len,), dtype=Observation)
        self.weights = torch.ones(self.max_len)
        self._ptr = 0
        self.transformations = transformations or list()
        self.num_steps = num_steps
        self.new_observation = True

    @classmethod
    def from_other(cls, other):
        """Create a Experience Replay from another one.

        All observations will be added sequentially, but only that will be copied.
        Weights will be initialized as if these were new observations.
        """
        new = cls(other.max_len, other.transformations, other.n_step)

        for observation in other.memory:
            if isinstance(observation, Observation):
                new.append(observation)
        return new

    def __len__(self):
        """Return the current size of the buffer."""
        if self.is_full:
            return self.max_len
        else:
            return self._ptr

    def __getitem__(self, idx):
        """Return any desired observation.

        Parameters
        ----------
        idx: int

        Returns
        -------
        observation: Observation
        idx: int
        weight: torch.tensor.

        """
        return self.get_observation(idx), idx, self.weights[idx]

    def get_observation(self, idx):
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

        if self.new_observation:
            observation = Observation(*[o.unsqueeze(0) for o in observation])
            self.new_observation = False
        else:
            observation = concatenate_observations(self.memory[self._ptr], observation)

        self.memory[self._ptr] = observation

        if self.memory[self._ptr].state.shape[0] == self.num_steps or \
                self.memory[self._ptr].done[-1]:
            self._ptr = (self._ptr + 1) % self.max_len
            self.new_observation = True

        for transformation in self.transformations:
            transformation.update(observation)

    def get_batch(self, batch_size):
        """Get a batch of data."""
        indices = np.random.choice(len(self), batch_size)
        return default_collate([self[i] for i in indices])

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
        all_obs = stack_list_of_tuples(self.memory[:self._ptr])

        for transformation in self.transformations:
            all_obs = transformation(all_obs)
        return all_obs

    def update(self, indexes, td_error):
        """Update experience replay sampling distribution with set of weights."""
        pass
