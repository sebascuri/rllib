import numpy as np
from rllib.dataset import Observation


class ExperienceReplay(object):
    """Experience Replay stores observations in a buffer of size `max_len'.

    Once the buffer is full, it forgets older observations. To sample an observation
    call the `sample' method and it sample observations i.i.d..

    The public methods are:
        __len__
        append
        sample
        is_full
    """
    def __init__(self, max_len, seed=None):
        """Initialize experience replay buffer.

        Parameters
        ----------
        max_len: int
        seed: int, optional

        """
        self._pointer = 0
        self._max_len = max_len
        self._memory = np.empty((self._max_len,), dtype=Observation)
        self._random = np.random.RandomState(seed)

    def __len__(self):
        """Current buffer size.

        Returns
        -------
        length: int
        """
        if self.is_full:
            return self._max_len
        else:
            return self._pointer

    def append(self, observation):
        """Append a new observation to the buffer.

        Parameters
        ----------
        observation: Observation

        Returns
        -------
        None

        """
        self._memory[self._pointer] = observation
        self._pointer = (self._pointer + 1) % self._max_len

    def sample(self, batch_size):
        """Sample i.i.d. a batch of observations from the buffer

        Parameters
        ----------
        batch_size: int

        Returns
        -------
        Observations batch:
        """
        """
        Sample i.i.d. a batch of observations from the buffer.

        :param batch_size: (int) size of batch to sample.

        :return Observation batch: (np-array of Observations).
        """
        if self.is_full:
            memory = self._memory
        else:
            memory = self._memory[:self._pointer]  # self._pointer is not included.
        return self._random.choice(memory, batch_size)

    @property
    def is_full(self):
        """
        Check if the memory is full.

        :return: (bool).
        """
        return self._memory[-1] is not None  # check if the last element is not empty.
