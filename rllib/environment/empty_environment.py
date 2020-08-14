"""Interface for Environments."""

import numpy as np
from gym.spaces import Box, Discrete

from .abstract_environment import AbstractEnvironment


class EmptyEnvironment(AbstractEnvironment):
    """Dummy Environment for testing."""

    def __init__(self, dim_state, dim_action, num_states=-1, num_actions=-1):
        if num_actions >= 0:
            action_space = Discrete(num_actions)
        else:
            action_space = Box(
                np.array([-1] * dim_action[0]), np.array([1] * dim_action[0])
            )

        if num_states >= 0:
            state_space = Discrete(num_states)
        else:
            state_space = Box(
                np.array([-1] * dim_state[0]), np.array([1] * dim_state[0])
            )

        super().__init__(
            dim_state,
            dim_action,
            state_space,
            action_space,
            num_states=num_states,
            num_actions=num_actions,
        )

    def step(self, action):
        """Run one time-step of the model dynamics.

        Parameters
        ----------
        action: ndarray

        Returns
        -------
        observation: ndarray
        reward: float
        done: bool
        info: dict

        """
        return self.observation_space.sample(), 0, False, {}

    def reset(self):
        """Reset the state of the model and returns an initial observation.

        Returns
        -------
        observation: ndarray

        """
        return self.observation_space.sample()

    @property
    def state(self):
        """Return current state of environment."""
        return self.observation_space.sample()

    @state.setter
    def state(self, value):
        """Set a place holder."""
        # This method has to be empty because there is no state.
        pass

    @property
    def time(self):
        """Return current time of environment."""
        return 0

    @property
    def name(self):
        """Return class name."""
        return self.__class__.__name__
