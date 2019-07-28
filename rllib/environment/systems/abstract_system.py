from abc import ABC, abstractmethod
from gym import spaces
import numpy as np


class AbstractSystem(ABC):
    def __init__(self, dim_state, dim_action, dim_observation=None):
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        if dim_observation is None:
            dim_observation = dim_state
        self.dim_observation = dim_observation
        self._time = 0

    @property
    @abstractmethod
    def state(self):
        raise NotImplementedError

    @state.setter
    @abstractmethod
    def state(self, value):
        raise NotImplementedError

    @property
    def time(self):
        return self._time

    @abstractmethod
    def step(self, action):
        raise NotImplementedError

    @abstractmethod
    def reset(self, state):
        raise NotImplementedError

    @property
    def action_space(self):
        return spaces.Box(np.array([-1e10] * self.dim_action),
                          np.array([1e10] * self.dim_action))

    @property
    def observation_space(self):
        return spaces.Box(np.array([-1e10] * self.dim_observation),
                          np.array([1e10] * self.dim_observation))
