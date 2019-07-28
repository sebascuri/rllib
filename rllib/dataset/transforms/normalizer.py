from .abstract_transform import AbstractTransform
from .. import Observation
import numpy as np


__all__ = ['StateNormalizer', 'ActionNormalizer']


def running_statistics(old_mean, old_var, old_count, new_mean, new_var, new_count):
    """Update mean and variance statistics based on a new batch of data.

    Uses a modified version of Welford's algorithm, see
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    Parameters
    ----------
    old_mean : ndarray
    old_var : ndarray
    old_count : int
    new_mean : ndarray
    new_var : ndarray
    new_count : int
    """
    delta = new_mean - old_mean
    total = old_count + new_count
    m_a = old_var * old_count
    m_b = new_var * new_count
    m_2 = m_a + m_b + delta ** 2 * new_count * (old_count / total)
    var = m_2 / total
    mean = old_mean + delta * (new_count / total)
    return mean, var


def normalize(array, mean, variance, preserve_origin=False):
    """Normalize an array.

    Parameters
    ----------
    array : ndarray
    mean : ndarray
    variance : ndarray
    preserve_origin : bool, optional
        Whether to retain the origin (sign) of the data.
    """
    if preserve_origin:
        scale = np.sqrt(variance + mean ** 2)
        return array / scale
    else:
        return (array - mean) / np.sqrt(variance)


def denormalize(array, mean, variance, preserve_origin=False):
    """Denormalize an array.

    Parameters
    ----------
    array : ndarray
    mean : ndarray
    variance : ndarray
    preserve_origin : bool, optional
        Whether to retain the origin (sign) of the data.
    """
    if preserve_origin:
        scale = np.sqrt(variance + mean ** 2)
        return array * scale
    else:
        return mean + array * np.sqrt(variance)


class Normalizer(object):
    def __init__(self, preserve_origin=False):
        super().__init__()
        self._mean = np.array(0.)
        self._variance = np.array(1.)
        self._count = 0
        self._preserve_origin = preserve_origin

    def __call__(self, array):
        return normalize(array, self._mean, self._variance, self._preserve_origin)

    def update(self, array):
        new_mean = np.mean(array, axis=0)
        new_var = np.var(array, axis=0)

        self._mean, self._variance = running_statistics(
            self._mean, self._variance, self._count, new_mean, new_var, len(array))

        self._count += len(array)

    def normalize(self, array):
        return normalize(array, self._mean, self._variance, self._preserve_origin)

    def denormalize(self, array):
        return denormalize(array, self._mean, self._variance, self._preserve_origin)


class StateNormalizer(AbstractTransform):
    def __init__(self, preserve_origin=False):
        super().__init__()
        self._normalizer = Normalizer(preserve_origin)

    def update(self, trajectory):
        self._normalizer.update(trajectory.state)

    def __call__(self, observation):
        return Observation(
            state=self._normalizer(observation.state),
            action=observation.action,
            reward=observation.reward,
            next_state=self._normalizer(observation.next_state),
            done=observation.done
        )

    def reverse(self, observation):
        return Observation(
            state=self._normalizer.denormalize(observation.state),
            action=observation.action,
            reward=observation.reward,
            next_state=self._normalizer.denormalize(observation.next_state),
            done=observation.done
        )


class ActionNormalizer(AbstractTransform):
    def __init__(self, preserve_origin=False):
        super().__init__()
        self._normalizer = Normalizer(preserve_origin)

    def update(self, trajectory):
        self._normalizer.update(trajectory.action)

    def __call__(self, observation):
        return Observation(
            state=observation.state,
            action=self._normalizer(observation.action),
            reward=observation.reward,
            next_state=observation.next_state,
            done=observation.done
        )

    def reverse(self, observation):
        return Observation(
            state=observation.state,
            action=self._normalizer.denormalize(observation.action),
            reward=observation.reward,
            next_state=observation.next_state,
            done=observation.done
        )
