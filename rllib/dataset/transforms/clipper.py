"""Implementation of a Transformation that clips attributes."""

import numpy as np
import torch
import torch.jit
import torch.nn as nn

from .abstract_transform import AbstractTransform


class Clipper(nn.Module):
    """Clipper Class."""

    def __init__(self, min_val, max_val):
        super().__init__()
        self._min = min_val
        self._max = max_val

    def forward(self, array):
        """See `AbstractTransform.__call__'."""
        if isinstance(array, torch.Tensor):
            return torch.clamp(array, self._min, self._max)
        else:
            return np.clip(array, self._min, self._max)

    @torch.jit.export
    def inverse(self, array):
        """See `AbstractTransform.inverse'."""
        return array


class RewardClipper(AbstractTransform):
    """Implementation of a Reward Clipper.

    Given a reward, it will clip it between min_reward and max_reward.

    Parameters
    ----------
    min_reward: float, optional (default=0.)
        minimum bound for rewards.

    max_reward: float, optional (default=1.)
        maximum bound for rewards.

    Notes
    -----
    This transformation does not have a inverse so the same observation is returned.

    """

    def __init__(self, min_reward=0.0, max_reward=1.0):
        super().__init__()
        self._clipper = Clipper(min_reward, max_reward)

    def forward(self, observation):
        """See `AbstractTransform.__call__'."""
        observation.reward = self._clipper(observation.reward)
        return observation

    @torch.jit.export
    def inverse(self, observation):
        """See `AbstractTransform.inverse'."""
        observation.reward = self._clipper.inverse(observation.reward)
        return observation


class ActionClipper(AbstractTransform):
    """Implementation of a Action Clipper.

    Given an action, it will clip it between min_action and max_action.

    Parameters
    ----------
    min_action: float, optional (default=0.)
        minimum bound for rewards.

    max_action: float, optional (default=1.)
        maximum bound for rewards.

    Notes
    -----
    This transformation does not have a inverse so the same observation is returned.

    """

    def __init__(self, min_action=-1.0, max_action=1.0):
        super().__init__()
        self._clipper = Clipper(min_action, max_action)

    def forward(self, observation):
        """See `AbstractTransform.__call__'."""
        observation.action = self._clipper(observation.action)
        return observation

    @torch.jit.export
    def inverse(self, observation):
        """See `AbstractTransform.inverse'."""
        observation.action = self._clipper.inverse(observation.action)
        return observation
