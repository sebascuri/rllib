"""Implementation of a Transformation that offsets the data with a mean function."""

from .abstract_transform import AbstractTransform
from .. import Observation

__all__ = ['MeanFunction']


class MeanFunction(AbstractTransform):
    """Implementation of a Mean function Clipper.

    Given a mean function, it will substract it from the next state.

    Parameters
    ----------
    mean_function : callable
        A callable that, given the current state and action, returns prediction for the
        `next_state`.
    """

    def __init__(self, mean_function):
        self.mean_function = mean_function

    def __call__(self, observation):
        """See `AbstractTransform.__call__'."""
        predicted_next_state = self.mean_function(observation.state, observation.action)
        return Observation(
            state=observation.state,
            action=observation.action,
            reward=observation.reward,
            next_state=observation.next_state - predicted_next_state,
            done=observation.done
        )

    def inverse(self, observation):
        """See `AbstractTransform.inverse'."""
        predicted_next_state = self.mean_function(observation.state, observation.action)
        return Observation(
            state=observation.state,
            action=observation.action,
            reward=observation.reward,
            next_state=observation.next_state + predicted_next_state,
            done=observation.done
        )
