"""Implementation of a Transformation that normalizes a vector."""

from .abstract_transform import AbstractTransform
from .. import Observation
from .utilities import Normalizer

__all__ = ['StateNormalizer', 'ActionNormalizer']


class StateNormalizer(AbstractTransform):
    """Implementation of a transformer that normalizes the observed (next) states.

    The state and next state of an observation are shifted by the mean and then are
    re-scaled with the standard deviation as:
        state = (raw_state - mean) / std_dev
        next_state = (raw_next_state - mean) / std_dev

    The mean and standard deviation are computed with running statistics of the action.

    Parameters
    ----------
    preserve_origin: bool, optional (default=False)
        preserve the origin when rescaling.

    """

    def __init__(self, preserve_origin=False):
        super().__init__()
        self._normalizer = Normalizer(preserve_origin)

    def __call__(self, observation):
        return Observation(
            state=self._normalizer(observation.state),
            action=observation.action,
            reward=observation.reward,
            next_state=self._normalizer(observation.next_state),
            done=observation.done
        )

    def inverse(self, observation):
        return Observation(
            state=self._normalizer.inverse(observation.state),
            action=observation.action,
            reward=observation.reward,
            next_state=self._normalizer.inverse(observation.next_state),
            done=observation.done
        )

    def update(self, observation):
        self._normalizer.update(observation.state)


class ActionNormalizer(AbstractTransform):
    """Implementation of a transformer that normalizes the observed action.

    The action of an observation is shifted by the mean and then re-scaled with the
    standard deviation as:
        action = (raw_action - mean) / std_dev

    The mean and standard deviation are computed with running statistics of the action.

    Parameters
    ----------
    preserve_origin: bool, optional (default=False)
        preserve the origin when rescaling.

    """

    def __init__(self, preserve_origin=False):
        super().__init__()
        self._normalizer = Normalizer(preserve_origin)

    def update(self, observation):
        self._normalizer.update(observation.action)

    def __call__(self, observation):
        return Observation(
            state=observation.state,
            action=self._normalizer(observation.action),
            reward=observation.reward,
            next_state=observation.next_state,
            done=observation.done
        )

    def inverse(self, observation):
        return Observation(
            state=observation.state,
            action=self._normalizer.inverse(observation.action),
            reward=observation.reward,
            next_state=observation.next_state,
            done=observation.done
        )
