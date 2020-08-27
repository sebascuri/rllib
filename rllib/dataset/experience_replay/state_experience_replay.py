"""Implementation of an Experience Replay Buffer."""

import numpy as np
import torch
from torch.utils import data
from torch.utils.data._utils.collate import default_collate


class StateExperienceReplay(data.Dataset):
    """A State distribution Experience Replay buffer.

    The Experience Replay algorithm stores states and access them IID. It has a
    size, and it erases the older samples, once the buffer is full, like on a queue.

    Parameters
    ----------
    max_len: int.
        buffer size of experience replay algorithm.

    Methods
    -------
    append(state) -> None:
        append an state to the dataset.
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

    """

    def __init__(self, max_len, dim_state):
        super().__init__()
        self.max_len = max_len
        self.dim_state = dim_state
        self.memory = torch.empty(
            (self.max_len,) + self.dim_state, dtype=torch.get_default_dtype()
        )
        self._ptr = 0
        self.is_full = False

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
        return self.memory[idx]

    def reset(self):
        """Reset memory to empty."""
        self.memory = torch.empty(
            (self.max_len,) + self.dim_state, dtype=torch.get_default_dtype()
        )
        self._ptr = 0
        self.is_full = False

    def append(self, state):
        """Append new observation to the dataset.

        Parameters
        ----------
        state: Tensor

        Raises
        ------
        TypeError
            If the new observation is not of type Observation.
        """
        assert state.dim() == 2
        num_states, dim_state = state.shape
        assert (dim_state,) == self.dim_state
        if num_states + self._ptr < self.max_len:
            self.memory[self._ptr : self._ptr + num_states] = state
            self._ptr += num_states
        else:
            self.is_full = True
            delta = num_states + self._ptr - self.max_len
            self.memory[self._ptr :] = state[delta:]
            self.memory[:delta] = state[:delta]
            self._ptr = delta

    def sample_batch(self, batch_size):
        """Get a batch of data."""
        indices = np.random.choice(len(self), batch_size)
        return default_collate([self[i] for i in indices])
