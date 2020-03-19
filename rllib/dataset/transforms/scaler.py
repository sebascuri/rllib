"""Implementation of a Transformation that scales attributes."""

import torch.nn as nn

from .abstract_transform import AbstractTransform


class Scaler(nn.Module):
    """Scaler Class."""

    def __init__(self, scale):
        super().__init__()
        self._scale = scale
        assert self._scale > 0, "Scale must be positive."

    def forward(self, array):
        """See `AbstractTransform.__call__'."""
        return array / self._scale

    def inverse(self, array):
        """See `AbstractTransform.inverse'."""
        return array * self._scale


class RewardScaler(AbstractTransform):
    """Implementation of a Reward Scaler.

    Given a reward, it will scale it by dividing it by scale.

    Parameters
    ----------
    scale: float.
    """

    def __init__(self, scale):
        super().__init__()
        self._scaler = Scaler(scale)

    def forward(self, observation):
        """See `AbstractTransform.__call__'."""
        return observation._replace(reward=self._scaler(observation.reward))

    def inverse(self, observation):
        """See `AbstractTransform.inverse'."""
        return observation._replace(reward=self._scaler.inverse(observation.reward))


class ActionScaler(AbstractTransform):
    """Implementation of an Action Scaler.

    Given an action, it will scale it by dividing it by scale.

    Parameters
    ----------
    scale: float.

    """

    def __init__(self, scale):
        super().__init__()
        self._scaler = Scaler(scale)

    def forward(self, observation):
        """See `AbstractTransform.__call__'."""
        return observation._replace(action=self._scaler(observation.action))

    def inverse(self, observation):
        """See `AbstractTransform.inverse'."""
        return observation._replace(action=self._scaler.inverse(observation.action))
