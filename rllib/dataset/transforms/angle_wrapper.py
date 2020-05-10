"""Implementation of a that wraps angles to -pi and pi."""

import torch
import torch.jit

from .abstract_transform import AbstractTransform
from rllib.dataset.datatypes import Observation


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

    def forward(self, observation: Observation):
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

        return Observation(
            state=state,
            action=observation.action,
            reward=observation.reward,
            next_state=next_state,
            done=observation.done,
            next_action=observation.next_action,
            log_prob_action=observation.log_prob_action,
            entropy=observation.entropy,
            state_scale_tril=observation.state_scale_tril,
            next_state_scale_tril=observation.next_state_scale_tril
        )

    @torch.jit.export
    def inverse(self, observation: Observation):
        """See `AbstractTransform.inverse'."""
        return observation
