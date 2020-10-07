"""Implementation of an Experience Replay Buffer."""
import math
import warnings
from dataclasses import asdict

import numpy as np
import torch
from torch.utils import data
from torch.utils.data._utils.collate import default_collate

from rllib.dataset.datatypes import Observation
from rllib.dataset.utilities import stack_list_of_tuples


class ExperienceReplay(data.Dataset):
    """An Experience Replay Buffer dataset.

    The Experience Replay algorithm stores transitions and access them IID.
    It erases the older samples once the buffer is full, like on a queue.

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
    append(observation) -> None:
        append an observation to the dataset.
    is_full: bool
        check if buffer is full.
    update(indexes, td_error):
        update experience replay sampling distribution with td_error feedback.
    all_data:
        Get all the transformed data.
    sample_batch(batch_size):
        Get a batch of data.
    reset():
        Reset the memory to zero.
    get_observation(idx):
        Get the observation at a given index.

    References
    ----------
    Lin, L. J. (1992).
    Self-improving reactive agents based on reinforcement learning, planning and
    teaching. Machine learning.

    TODO: Make this class robust, easy to use, and fast.
    """

    def __init__(self, max_len, transformations=None, num_steps=0):
        super().__init__()
        self.max_len = max_len
        self.memory = np.empty((self.max_len,), dtype=Observation)

        self.valid = torch.zeros(self.max_len)
        self.weights = torch.ones(self.max_len)
        self.data_count = 0

        self.transformations = transformations or list()
        self._num_steps = num_steps
        self.zero_observation = None

        self.raw = False

        if self.num_steps < 0:
            raise ValueError("Number of steps must be non-negative.")

    @classmethod
    def from_other(cls, other, num_steps=None):
        """Create a Experience Replay from another one.

        All observations will be added sequentially, but only that will be copied.
        Weights will be initialized as if these were new observations.
        """
        num_steps = other.num_steps if num_steps is None else num_steps
        new = cls(other.max_len, other.transformations, num_steps)

        start_idx = other.ptr
        for i in range(other.max_len):
            # Start iterating at the next location.
            # In this case, when the num_steps change, old observations are erased.

            if other.valid[(start_idx + i) % other.max_len]:
                observation = other.memory[(start_idx + i) % other.max_len]
                new.append(observation)
            elif other.valid[(start_idx + i - 1) % other.max_len]:  # Last of episode.
                new.end_episode()
            else:  # Either empty or belongs to some padding.
                pass

        return new

    def split(self, ratio=0.8, *args, **kwargs):
        """Split into two data sets."""
        idx = np.arange(0, len(self))
        np.random.shuffle(idx)
        split_idx = math.ceil(ratio * len(self))
        train_idx = idx[:split_idx]
        test_idx = idx[split_idx:]

        train = type(self)(
            max_len=self.max_len, transformations=self.transformations, *args, **kwargs
        )
        test = type(self)(
            max_len=self.max_len, transformations=self.transformations, *args, **kwargs
        )

        for dataset, idx in zip([train, test], [train_idx, test_idx]):
            for i in idx:
                dataset.memory[i] = self.memory[i]
                dataset.valid[i] = self.valid[i]
                dataset.weights[i] = self.weights[i]
                dataset.data_count += 1

        return train, test

    def __len__(self):
        """Return the current size of the buffer."""
        if self.is_full:
            return self.max_len
        else:
            return self.ptr

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
        if self.valid[idx] == 0:  # when a non-valid index is sampled.
            idx = np.random.choice(self.valid_indexes).item()

        return asdict(self._get_observation(idx)), idx, self.weights[idx]

    def _init_observation(self, observation):
        if observation.state.ndim == 0:
            dim_state, num_states = 1, 1
        else:
            dim_state, num_states = observation.state.shape[-1], -1

        if observation.action.ndim == 0:
            dim_action, num_actions = 1, 1
        else:
            dim_action, num_actions = observation.action.shape[-1], -1

        self.zero_observation = Observation.zero_example(
            dim_state=dim_state,
            dim_action=dim_action,
            num_states=num_states,
            num_actions=num_actions,
        )

    def _get_consecutive_observations(self, start_idx, num_steps):
        if num_steps == 0 and not (
            isinstance(start_idx, int) or isinstance(start_idx, np.int)
        ):
            observation = stack_list_of_tuples(self.memory[start_idx])
            return Observation(*map(lambda x: x.unsqueeze(1), observation))
        num_steps = max(1, num_steps)
        if start_idx + num_steps <= self.max_len:
            obs_list = self.memory[start_idx : start_idx + num_steps]
        else:  # The trajectory is split by the circular buffer.
            delta_idx = start_idx + num_steps - self.max_len
            obs_list = np.concatenate(
                (self.memory[start_idx : self.max_len], self.memory[:delta_idx])
            )

        return stack_list_of_tuples(obs_list)

    def _get_observation(self, idx):
        """Return any desired observation.

        Parameters
        ----------
        idx: int

        Returns
        -------
        observation: Observation

        """
        observation = self._get_consecutive_observations(idx, self.num_steps)
        if self.raw:
            return observation
        for transform in self.transformations:
            observation = transform(observation)
        return observation

    def reset(self):
        """Reset memory to empty."""
        self.memory = np.empty((self.max_len,), dtype=Observation)
        self.valid = torch.zeros(self.max_len)
        self.data_count = 0
        self.zero_observation = None

    def end_episode(self):
        """Terminate an episode.

        It appends `num_steps' invalid transitions to the replay buffer.
        """
        for _ in range(self.num_steps):
            self.data_count += 1

    def append_invalid(self):
        """Append an invalid transition."""
        if self.zero_observation is None:
            warnings.warn("Buffer not initialized.", RuntimeWarning)
        else:
            self.memory[self.ptr] = self.zero_observation
            self.valid[self.ptr] = 0
            self.data_count += 1

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
        if not isinstance(observation, Observation):
            raise TypeError(
                f"input has to be of type Observation, and it was {type(observation)}"
            )

        if self.zero_observation is None:
            self._init_observation(observation)

        self.memory[self.ptr] = observation.clone()
        self.valid[self.ptr] = 1

        for i in range(self.num_steps):
            self.memory[(self.ptr + i + 1) % self.max_len] = self.zero_observation
            self.valid[(self.ptr + i + 1) % self.max_len] = 0
        self.data_count += 1

        for transformation in self.transformations:
            transformation.update(observation)
            observation = transformation(observation)

    def sample_batch(self, batch_size):
        """Sample a batch of observations."""
        indices = np.random.choice(self.valid_indexes, batch_size)
        if self.num_steps == 0:
            obs = self._get_observation(indices)
            return (obs, torch.tensor(indices), self.weights[indices])
        else:
            obs, idx, weight = default_collate([self[i] for i in indices])
            return Observation(**obs), idx, weight

    @property
    def is_full(self):
        """Flag that checks if memory in buffer is full.

        Returns
        -------
        bool
        """
        return self.data_count >= self.max_len

    @property
    def all_data(self):
        """Get all the data."""
        all_obs = self.all_raw

        for transformation in self.transformations:
            all_obs = transformation(all_obs)
        return all_obs

    @property
    def all_raw(self):
        """Get all the un-transformed data."""
        all_raw = stack_list_of_tuples(self.memory[self.valid_indexes])
        return all_raw

    @property
    def ptr(self):
        """Return data pointer where the next transition will be written."""
        return self.data_count % self.max_len

    @property
    def valid_indexes(self):
        """Return list of valid indexes."""
        return torch.nonzero(self.valid, as_tuple=False).squeeze(1)

    @property
    def num_steps(self):
        """Return the number of steps."""
        return self._num_steps

    @num_steps.setter
    def num_steps(self, value):
        """Reset the number of steps."""
        self._num_steps = value
        other = ExperienceReplay.from_other(self, num_steps=value)
        self.memory = other.memory
        self.valid = other.valid
        self.data_count = other.data_count
        self.weights = other.weights

    def update(self, indexes, td_error):
        """Update experience replay sampling distribution with set of weights."""
        pass
