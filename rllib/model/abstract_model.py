"""Interface for dynamical models."""

from abc import ABCMeta, abstractmethod


class AbstractModel(object, metaclass=ABCMeta):
    """Interface for Models of an Environment.

    A Model is an approximation of the environment.
    As such it has a step method that returns a `Distribution' over next states,
    instead of the next state.

    Parameters
    ----------
    dim_state: int
        dimension of state.
    dim_action: int
        dimension of action.
    dim_observation: int
        dimension of observation.
    num_states: int, optional
        number of discrete states (None if state is continuous).
    num_actions: int, optional
        number of discrete actions (None if action is continuous).
    num_observations: int, optional
        number of discrete observations (None if observation is continuous).

    Methods
    -------
    __call__(state, action): torch.Distribution
        return the next state distribution given a state and an action.
    reward(state, action): float
        return the reward the model predicts.
    initial_state: torch.Distribution
        return the initial state distribution.

    discrete_state: bool
        Flag that indicates if state space is discrete.
    discrete_action: bool
        Flag that indicates if action space is discrete.
    discrete_observation: bool
        Flag that indicates if observation space is discrete.

    """

    def __init__(self, dim_state, dim_action, dim_observation=None, num_states=None,
                 num_actions=None, num_observations=None):
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.dim_observation = dim_observation if dim_observation else dim_state

        self.num_states = num_states
        self.num_actions = num_actions
        self.num_observations = num_observations

    @abstractmethod
    def __call__(self, state, action):
        """Get next-state distribution.

        Parameters
        ----------
        state: array_like
        action: array_like

        Returns
        -------
        next-state: torch.distributions.Distribution

        """
        raise NotImplementedError

    @property
    def parameters(self):
        """Get model parameters."""
        return None

    @parameters.setter
    def parameters(self, new_value):
        pass

    @property
    def discrete_state(self):
        """Check if state space is discrete."""
        return self.num_states is None

    @property
    def discrete_action(self):
        """Check if action space is discrete."""
        return self.num_actions is None

    @property
    def discrete_observation(self):
        """Check if observation space is discrete."""
        return self.num_observations is None
