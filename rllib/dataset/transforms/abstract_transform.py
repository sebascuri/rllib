"""Interface for Transformers of a dataset."""

from abc import ABCMeta

import torch.jit
import torch.nn as nn

from rllib.dataset.datatypes import Observation


class AbstractTransform(nn.Module, metaclass=ABCMeta):
    """Abstract transform to apply on a dataset.

    Methods
    -------
    __call__(observation): Observation
        transform a raw observation.
    inverse(observation): Observation
        revert the transformation of the observation.
    update(observation):
        update the parameters of the transformer.

    """

    def forward(self, observation: Observation):
        """Apply transformation to observation tuple.

        Parameters
        ----------
        observation: Observation
            Raw observation.

        Returns
        -------
        observation: Observation
            Transformed observation.
        """
        raise NotImplementedError

    @torch.jit.export
    def inverse(self, observation: Observation):
        """Apply the inverse transformation to observation tuple..

        Parameters
        ----------
        observation: Observation
            Transformed observation.

        Returns
        -------
        observation: Observation
            Inverse-transformed observation.

        """
        return observation

    @torch.jit.export
    def update(self, observation: Observation):
        """Update parameters of transformer.

        Parameters
        ----------
        observation: Observation

        """
        pass
