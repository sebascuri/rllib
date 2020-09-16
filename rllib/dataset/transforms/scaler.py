"""Implementation of a Transformation that scales attributes."""

import torch.jit
import torch.nn as nn

from .abstract_transform import AbstractTransform


class Scaler(nn.Module):
    """Scaler Class."""

    def __init__(self, scale):
        super().__init__()
        if not isinstance(scale, torch.Tensor):
            self._scale = torch.tensor(scale, dtype=torch.get_default_dtype())
        self._scale[self._scale == 0] = 1.0
        assert torch.all(self._scale > 0), "Scale must be positive."

    def forward(self, array):
        """See `AbstractTransform.__call__'."""
        return array / self._scale

    @torch.jit.export
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
        observation.reward = self._scaler(observation.reward)
        return observation

    @torch.jit.export
    def inverse(self, observation):
        """See `AbstractTransform.inverse'."""
        observation.reward = self._scaler.inverse(observation.reward)
        return observation


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
        observation.action = self._scaler(observation.action)
        return observation

    @torch.jit.export
    def inverse(self, observation):
        """See `AbstractTransform.inverse'."""
        observation.action = self._scaler.inverse(observation.action)
        return observation
