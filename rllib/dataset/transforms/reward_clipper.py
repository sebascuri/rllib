from .abstract_transform import AbstractTransform
from ..observation import Observation
import numpy as np


class RewardClipper(AbstractTransform):
    def __init__(self, min_reward=0, max_reward=1):
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

    def reverse(self, observation):
        return observation
