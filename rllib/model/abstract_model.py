"""Interface for dynamical models."""

from abc import ABC


class AbstractModel(ABC):
    """Interface for Models of an Environment.

    A Model is an approximation of the environment.
    As such it has a step method that returns a `Distribution' over next states,
    instead of the next state.

    Parameters
    ----------
    dim_state: int
        dimension of state.
    num_states: int, optional
        number of discrete states (None if state is continuous).

    Attributes
    ----------
    dim_state: int
        dimension of state.
    num_states: int
        number of discrete states (None if state is continuous).

    Methods
    -------
    __call__(state, action=None): float
        return the value of a given state.
    parameters: generator
        return the value function parametrization.
    discrete_state: bool
        Flag that indicates if state space is discrete.

    """

    def __init__(self, dim_state, dim_action, dim_observation=None, num_state=None,
                 num_actions=None, num_observations=None):
        pass
