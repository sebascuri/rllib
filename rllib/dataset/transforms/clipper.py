"""Implementation of a Transformation that clips attributes."""

import numpy as np
import torch
import torch.jit
import torch.nn as nn

from rllib.dataset.datatypes import Observation

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

    def forward(self, observation: Observation):
        """See `AbstractTransform.__call__'."""
        return Observation(
            state=observation.state,
            action=observation.action,
            reward=self._clipper(observation.reward),
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
            reward=self._clipper.inverse(observation.reward),
            next_state=observation.next_state,
            done=observation.done,
            next_action=observation.next_action,
            log_prob_action=observation.log_prob_action,
            entropy=observation.entropy,
            state_scale_tril=observation.state_scale_tril,
            next_state_scale_tril=observation.next_state_scale_tril,
        )


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

    def forward(self, observation: Observation):
        """See `AbstractTransform.__call__'."""
        return Observation(
            state=observation.state,
            action=self._clipper(observation.action),
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
            action=self._clipper.inverse(observation.action),
            reward=observation.reward,
            next_state=observation.next_state,
            done=observation.done,
            next_action=observation.next_action,
            log_prob_action=observation.log_prob_action,
            entropy=observation.entropy,
            state_scale_tril=observation.state_scale_tril,
            next_state_scale_tril=observation.next_state_scale_tril,
        )
