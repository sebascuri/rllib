import numpy as np
from . import Observation
from torch.utils import data


class ExperienceReplay(data.Dataset):
    def __init__(self, max_len, transforms=None):
        self._max_len = max_len
        self._memory = np.empty((self._max_len,), dtype=Observation)
        self._sampling_idx = []
        self._ptr = 0
        if transforms is None:
            transforms = []
        self._transforms = transforms

    def __getitem__(self, idx):
        assert idx < len(self)
        idx = self._sampling_idx[idx]  # This is done for shuffling.
        """If memory was directly shuffled, then ordering would be lost.
        """

        observation = self._memory[idx]
        for transform in self._transforms:
            observation = transform(observation)
        return observation

    def __len__(self):
        if self.is_full:
            return self._max_len
        else:
            return self._ptr

    def append(self, observation):
        if not type(observation) == Observation:
            raise TypeError("""
            input has to be of type Observation, and it was found of type {}
            """.format(type(observation)))
        if not self.is_full:
            self._sampling_idx.append(self._ptr)
        self._memory[self._ptr] = observation
        self._ptr = (self._ptr + 1) % self._max_len

    @property
    def is_full(self):
        """
        Check if the memory is full.

        :return: (bool).
        """
        return self._memory[-1] is not None  # check if the last element is not empty.

    def shuffle(self):
        np.random.shuffle(self._sampling_idx)
