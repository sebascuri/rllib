"""Implementation of a that wraps angles to -pi and pi."""

import torch
import torch.jit

from rllib.dataset.datatypes import Observation

from .abstract_transform import AbstractTransform


class AngleWrapper(AbstractTransform):
    """Implementation of a Reward Clipper.

    Given a state vector, it will wrap the angles to a value between -pi and pi.

    Parameters
    ----------
    indexes: List[int].
        indexes where there are angles.


    """

    def __init__(self, indexes):
        super().__init__()
        self._indexes = indexes

    def forward(self, observation):
        """See `AbstractTransform.__call__'."""
        state = observation.state
        angles = state[..., self._indexes]
        cos, sin = torch.cos(angles), torch.sin(angles)
        state[..., self._indexes] = torch.atan2(sin, cos)

        next_state = observation.next_state
        if observation.next_state.dim() > 0:
            angles = next_state[..., self._indexes]
            cos, sin = torch.cos(angles), torch.sin(angles)
            next_state[..., self._indexes] = torch.atan2(sin, cos)

        observation.state = state
        observation.next_state = next_state
        return observation

    @torch.jit.export
    def inverse(self, observation: Observation):
        """See `AbstractTransform.inverse'."""
        return observation
