"""Interface for Environments."""


from abc import ABC, abstractmethod


class AbstractEnvironment(ABC):
    """Interface for Environments.

    Parameters
    ----------
    dim_state: int
        dimension of state.
    dim_action: int
        dimension of action.
    observation_space: gym.env.Spaces
    action_space: gym.env.Spaces
    dim_observation: int, optional
        dimension of observation.
    num_observations: int, optional
        number of discrete observations (None if observation is continuous).
    num_actions: int, optional
        number of discrete actions (None if action is continuous).

    Methods
    -------
    step(action): next_state, reward, done, info
        execute a step in the environment.
    reset(): reset the environment.

    """

    def __init__(self, dim_state, dim_action, observation_space, action_space,
                 dim_observation=None, num_states=None, num_actions=None,
                 num_observations=None):
        super().__init__()
        self.dim_action = dim_action
        self.dim_state = dim_state
        self.num_actions = num_actions
        self.num_observations = num_observations
        self.num_states = num_states

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
        """Return current state of environment."""
        raise NotImplementedError

    @state.setter
    @abstractmethod
    def state(self, value):
        raise NotImplementedError

    @property
    @abstractmethod
    def time(self):
        """Return current time of environment."""
        raise NotImplementedError

    @property
    def discrete_state(self):
        """Check if state space is discrete."""
        return self.num_states is not None

    @property
    def discrete_action(self):
        """Check if action space is discrete."""
        return self.num_actions is not None

    @property
    def discrete_observation(self):
        """Check if observation space is discrete."""
        return self.num_observations is not None
