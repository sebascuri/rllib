"""Interface for physical systems."""

from abc import ABCMeta, abstractmethod
from gym import spaces
import numpy as np


class AbstractSystem(object, metaclass=ABCMeta):
    """Interface for physical systems with continuous state-action spaces.

    Parameters
    ----------
    dim_state: int
    dim_action: int
    dim_observation: int, optional

    Methods
    -------
    state: ndarray
        return the current state of the system.
    time: int or float
        return the current time step.
    reset(state):
        reset the state.
    step(action): ndarray
        execute a one step simulation and return the next state.
    """

    def __init__(self, dim_state, dim_action, dim_observation=None):
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        if dim_observation is None:
            dim_observation = dim_state
        self.dim_observation = dim_observation
        self._time = 0

    @property  # type: ignore
    @abstractmethod
    def state(self):
        """Return the state of the system."""
        raise NotImplementedError

    @state.setter  # type: ignore
    @abstractmethod
    def state(self, value):
        raise NotImplementedError

    @property
    def time(self):
        """Return the current time of the system."""
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
        """Reset system and set state to `state'.

        Parameters
        ----------
        state: ndarray

        """
        raise NotImplementedError

    def render(self, mode='human'):
        """Render system."""
        pass

    @property
    def action_space(self):
        """Return action space."""
        return spaces.Box(np.array([-1e10] * self.dim_action),
                          np.array([1e10] * self.dim_action))

    @property
    def observation_space(self):
        """Return observation space."""
        return spaces.Box(np.array([-1e10] * self.dim_observation),
                          np.array([1e10] * self.dim_observation))
