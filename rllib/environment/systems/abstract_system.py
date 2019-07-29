from abc import ABC, abstractmethod
from gym import spaces
import numpy as np


class AbstractSystem(ABC):
    """Abstract System
    """
    def __init__(self, dim_state, dim_action, dim_observation=None):
        """Initialize an abstract system

        Parameters
        ----------
        dim_state: int
        dim_action: int
        dim_observation: int, optional
        """
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
        """Do a one step ahead simulation of the system.
        x' = f(x, action)

        Parameters
        ----------
        action: ndarray

        Returns
        -------
        next_state: ndarray
        """
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
