"""Implementation of a Transformation that offsets the data with a mean function."""

import torch.jit
import torch.nn as nn

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

    def forward(self, observation):
        """See `AbstractTransform.__call__'."""
        mean_next_state = self.mean_function(observation.state, observation.action)
        observation.next_state = observation.next_state - mean_next_state
        return observation

    @torch.jit.export
    def inverse(self, observation):
        """See `AbstractTransform.inverse'."""
        mean_next_state = self.mean_function(observation.state, observation.action)
        observation.next_state = observation.next_state + mean_next_state
        return observation
