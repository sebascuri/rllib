"""Implementation of a Transformation that offsets the data with a mean function."""

import torch.jit
import torch.nn as nn

from rllib.dataset.datatypes import Observation

from .abstract_transform import AbstractTransform


class DeltaState(nn.Module):
    """Implementation of a Mean function that returns f(s, a) = s."""

    def forward(self, state, action):
        """Compute next state."""
        return state


class MeanFunction(AbstractTransform):
    """Implementation of a Mean function Clipper.

    Given a mean function, it will substract it from the next state.

    Parameters
    ----------
    mean_function : nn.Module
        A nn.Module that, given the current state and action, returns prediction for the
        `next_state`.
    """

    def __init__(self, mean_function):
        super().__init__()
        self.mean_function = mean_function

    def forward(self, observation: Observation):
        """See `AbstractTransform.__call__'."""
        mean_next_state = self.mean_function(observation.state, observation.action)
        return Observation(
            state=observation.state,
            action=observation.action,
            reward=observation.reward,
            next_state=observation.next_state - mean_next_state,
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
        mean_next_state = self.mean_function(observation.state, observation.action)
        return Observation(
            state=observation.state,
            action=observation.action,
            reward=observation.reward,
            next_state=observation.next_state + mean_next_state,
            done=observation.done,
            next_action=observation.next_action,
            log_prob_action=observation.log_prob_action,
            entropy=observation.entropy,
            state_scale_tril=observation.state_scale_tril,
            next_state_scale_tril=observation.next_state_scale_tril,
        )
