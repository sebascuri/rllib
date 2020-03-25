"""Implementation of an Experience Replay Buffer with a Bootstrap mask."""


from .experience_replay import ExperienceReplay
import torch
from torch.distributions import Poisson
import numpy as np

from rllib.dataset.datatypes import Observation


class BootstrapExperienceReplay(ExperienceReplay):
    """An Bootstrap Experience Replay Buffer dataset.

    The BER stores transitions + a bootstrap mask and access them IID. It has a
    size, and it erases the older samples, once the buffer is full, like on a queue.

    The bootstrap distribution samples a mask according to a Poisson(1) distribution.
    The Poisson(1) distribution is an asymptotic approximation to Bin(N, 1/N).

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
    num_bootstraps: int, optional.
        Number of bootstrap data sets that the ER must maintain.

    References
    ----------
    Osband, I., Blundell, C., Pritzel, A., & Van Roy, B. (2016).
    Deep exploration via bootstrapped DQN. NeuRIPS.
    """

    def __init__(self, max_len, batch_size=1, transformations=None, num_bootstraps=1):
        super().__init__(max_len, batch_size, transformations)
        self.bootstrap_mask = np.empty((self.max_len, num_bootstraps))
        self.mask_distribution = Poisson(torch.ones(num_bootstraps))

    def __getitem__(self, idx):
        """Return any desired observation.

        Parameters
        ----------
        idx: int

        Returns
        -------
        observation: Observation

        """
        observation = super().__getitem__(idx)
        return observation, self.bootstrap_mask[idx]

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

        self.bootstrap_mask[self._ptr] = self.mask_distribution.sample()
        super().append(observation)
