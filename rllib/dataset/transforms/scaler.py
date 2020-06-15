"""Implementation of a Transformation that scales attributes."""

import torch.jit
import torch.nn as nn

from rllib.dataset.datatypes import Observation

from .abstract_transform import AbstractTransform


class Scaler(nn.Module):
    """Scaler Class."""

    def __init__(self, scale):
        super().__init__()
        self._scale = torch.tensor(scale, dtype=torch.get_default_dtype())
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

    def forward(self, observation: Observation):
        """See `AbstractTransform.__call__'."""
        return Observation(
            state=observation.state,
            action=observation.action,
            reward=self._scaler(observation.reward),
            next_state=observation.next_state,
            done=observation.done,
            next_action=observation.next_action,
            log_prob_action=observation.log_prob_action,
            entropy=observation.entropy,
            state_scale_tril=observation.state_scale_tril,
            next_state_scale_tril=observation.next_state_scale_tril,
        )

    @torch.jit.export
    def inverse(self, observation: Observation):
        """See `AbstractTransform.inverse'."""
        return Observation(
            state=observation.state,
            action=observation.action,
            reward=self._scaler.inverse(observation.reward),
            next_state=observation.next_state,
            done=observation.done,
            next_action=observation.next_action,
            log_prob_action=observation.log_prob_action,
            entropy=observation.entropy,
            state_scale_tril=observation.state_scale_tril,
            next_state_scale_tril=observation.next_state_scale_tril,
        )


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

    def forward(self, observation: Observation):
        """See `AbstractTransform.__call__'."""
        return Observation(
            state=observation.state,
            action=self._scaler(observation.action),
            reward=observation.reward,
            next_state=observation.next_state,
            done=observation.done,
            next_action=observation.next_action,
            log_prob_action=observation.log_prob_action,
            entropy=observation.entropy,
            state_scale_tril=observation.state_scale_tril,
            next_state_scale_tril=observation.next_state_scale_tril,
        )

    @torch.jit.export
    def inverse(self, observation: Observation):
        """See `AbstractTransform.inverse'."""
        return Observation(
            state=observation.state,
            action=self._scaler.inverse(observation.action),
            reward=observation.reward,
            next_state=observation.next_state,
            done=observation.done,
            next_action=observation.next_action,
            log_prob_action=observation.log_prob_action,
            entropy=observation.entropy,
            state_scale_tril=observation.state_scale_tril,
            next_state_scale_tril=observation.next_state_scale_tril,
        )
