"""Implementation of a Transformation that clips rewards."""

from .abstract_transform import AbstractTransform
from .. import Observation
import numpy as np

__all__ = ['RewardClipper']


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

    def __init__(self, min_reward=0., max_reward=1.):
        super().__init__()
        self._min_reward = min_reward
        self._max_reward = max_reward

    def __call__(self, observation):
        return Observation(state=observation.state,
                           action=observation.action,
                           next_state=observation.next_state,
                           done=observation.done,
                           reward=np.clip(observation.reward,
                                          self._min_reward,
                                          self._max_reward)
                           )

    def update(self, trajectory):
        pass

    def inverse(self, observation):
        return observation
