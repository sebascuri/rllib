"""Implementation of an Experience Replay Buffer with a Bootstrap mask."""

import torch
from torch.distributions import Poisson

from rllib.dataset.datatypes import Observation

from .experience_replay import ExperienceReplay


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

    def __init__(self, num_bootstraps=1, bootstrap=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = torch.empty(self.max_len, num_bootstraps, dtype=torch.int)
        self.mask_distribution = Poisson(torch.ones(num_bootstraps))
        self.bootstrap = bootstrap

    def append(self, observation):
        """Append new observation to the dataset.

        Every time a new observation is appended, sample a mask to build a Bootstrap.

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
                f"input has to be of type Observation, and it was {type(observation)}"
            )

        if self.bootstrap:
            self.weights[self.ptr] = self.mask_distribution.sample()
        else:
            self.weights[self.ptr] = torch.ones(self.mask_distribution.batch_shape)
        super().append(observation)

    def split(self, ratio=0.8, *args, **kwargs):
        """Split into two data sets."""
        return super().split(
            ratio=ratio,
            num_bootstraps=self.weights.shape[-1],
            bootstrap=self.bootstrap,
            *args,
            **kwargs,
        )
