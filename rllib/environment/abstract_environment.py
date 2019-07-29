"""Base class for environments."""
from abc import ABC, abstractmethod


class AbstractEnvironment(ABC):
    """The abstract Environment class encapsulates a Model of the environment.

    The public attributes are:
        action_space
        observation_space
    """
    def __init__(self, dim_state, dim_action, observation_space, action_space,
                 dim_observation=None, num_action=None, num_observation=None):
        """Initialize the environment

        Parameters
        ----------
        dim_state: int
        dim_action: int
        observation_space: gym.env.Spaces
        action_space: gym.env.Spaces
        dim_observation: int
        num_action: int
        num_observation: int
        """
        super().__init__()
        self.dim_action = dim_action
        self.dim_state = dim_state
        self.num_action = num_action
        self.num_observation = num_observation

        if dim_observation is None:
            dim_observation = dim_state
        self.dim_observation = dim_observation

        self.action_space = action_space
        self.observation_space = observation_space

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Reset the state of the model and returns an initial observation.

        Returns
        -------
        observation: ndarray

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def state(self):
        raise NotImplementedError

    @state.setter
    @abstractmethod
    def state(self, value):
        raise NotImplementedError

    @property
    @abstractmethod
    def time(self):
        raise NotImplementedError
