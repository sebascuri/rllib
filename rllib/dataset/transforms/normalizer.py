"""Implementation of a Transformation that normalizes attributes."""

from .abstract_transform import AbstractTransform
from rllib.dataset.transforms.utilities import get_backend, normalize, denormalize, \
    update_var, update_mean
import numpy as np


class Normalizer(object):
    """Normalizer Class."""

    def __init__(self, preserve_origin=False):
        super().__init__()
        self._mean = None
        self._variance = None
        self._count = 0
        self._preserve_origin = preserve_origin

    def __call__(self, array):
        """See `AbstractTransform.__call__'."""
        return normalize(array, self._mean, self._variance, self._preserve_origin)

    def inverse(self, array):
        """See `AbstractTransform.inverse'."""
        return denormalize(array, self._mean, self._variance, self._preserve_origin)

    def update(self, array):
        """See `AbstractTransform.update'."""
        backend = get_backend(array)
        if self._mean is None:
            self._mean = backend.zeros(1)
        if self._variance is None:
            self._variance = backend.ones(1)

        new_mean = backend.mean(array, 0)
        new_var = backend.var(array, 0)
        new_count = len(array)

        self._variance = update_var(self._mean, self._variance, self._count,
                                    new_mean, new_var, new_count,
                                    backend is np)
        self._mean = update_mean(self._mean, self._count, new_mean, new_count)

        self._count += new_count


class StateActionNormalizer(AbstractTransform):
    r"""Transformer that normalizes the states, next states, and actions.

    It compounds a StateNormalizer with an ActionNormalizer.

    Parameters
    ----------
    preserve_origin: bool, optional (default=False)
        preserve the origin when rescaling.

    """

    def __init__(self, preserve_origin=False):
        super().__init__()
        self._state_normalizer = StateNormalizer(preserve_origin)
        self._action_normalizer = ActionNormalizer(preserve_origin)

    def __call__(self, observation):
        """See `AbstractTransform.__call__'."""
        return self._action_normalizer(self._state_normalizer(observation))

    def inverse(self, observation):
        """See `AbstractTransform.inverse'."""
        return self._action_normalizer.inverse(
            self._state_normalizer.inverse(observation))

    def update(self, observation):
        """See `AbstractTransform.update'."""
        self._state_normalizer.update(observation)
        self._action_normalizer.update(observation)


class StateNormalizer(AbstractTransform):
    r"""Implementation of a transformer that normalizes the states and next states.

    The state and next state of an observation are shifted by the mean and then are
    re-scaled with the standard deviation as:
        .. math:: state = (raw_state - mean) / std_dev

    The mean and standard deviation are computed with running statistics of the state.

    Parameters
    ----------
    preserve_origin: bool, optional (default=False)
        preserve the origin when rescaling.

    """

    def __init__(self, preserve_origin=False):
        super().__init__()
        self._normalizer = Normalizer(preserve_origin)

    def __call__(self, observation):
        """See `AbstractTransform.__call__'."""
        return observation._replace(state=self._normalizer(observation.state),
                                    next_state=self._normalizer(observation.next_state))

    def inverse(self, observation):
        """See `AbstractTransform.inverse'."""
        return observation._replace(
            state=self._normalizer.inverse(observation.state),
            next_state=self._normalizer.inverse(observation.next_state))

    def update(self, observation):
        """See `AbstractTransform.update'."""
        self._normalizer.update(observation.state)


class ActionNormalizer(AbstractTransform):
    """Implementation of a transformer that normalizes the action.

    The action of an observation is shifted by the mean and then re-scaled with the
    standard deviation as:
        .. math:: action = (raw_action - mean) / std_dev

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
        """See `AbstractTransform.__call__'."""
        return observation._replace(action=self._normalizer(observation.action))

    def inverse(self, observation):
        """See `AbstractTransform.inverse'."""
        return observation._replace(action=self._normalizer.inverse(observation.action))

    def update(self, observation):
        """See `AbstractTransform.update'."""
        self._normalizer.update(observation.action)
