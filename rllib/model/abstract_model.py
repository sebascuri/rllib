"""Interface for dynamical models."""

from abc import ABCMeta
import torch.jit
import torch.nn as nn


class AbstractModel(nn.Module, metaclass=ABCMeta):
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

    def __init__(self, dim_state, dim_action, dim_observation=-1, num_states=-1,
                 num_actions=-1, num_observations=-1):
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.dim_observation = dim_observation if dim_observation else dim_state

        self.num_states = num_states if num_states is not None else -1
        self.num_actions = num_actions if num_actions is not None else -1
        self.num_observations = num_observations if num_observations is not None else -1

        self.discrete_state = self.num_states >= 0
        self.discrete_action = self.num_actions >= 0

    @property
    def name(self):
        """Get Model name."""
        return self.__class__.__name__
