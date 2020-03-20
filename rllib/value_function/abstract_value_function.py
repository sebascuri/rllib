"""Interface for value and q-functions."""

from abc import ABCMeta

import torch.nn as nn


class AbstractValueFunction(nn.Module, metaclass=ABCMeta):
    """Interface for Value Functions that describe the policy on an environment.

    A Value Function is a function that maps a state to a real value. This value is the
    expected sum of discounted returns that the agent will encounter by following the
    policy on the environment.

    Parameters
    ----------
    dim_state: int
        dimension of state.
    num_states: int, optional
        number of discrete states (None if state is continuous).
    tau: float (1.0)
        soft update parameter.

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

    def __init__(self, dim_state, num_states=-1, tau=1.0):
        super().__init__()
        self.dim_state = dim_state
        self.num_states = num_states if num_states is not None else -1
        self.tau = tau
        self.discrete_state = self.num_states >= 0


class AbstractQFunction(AbstractValueFunction):
    """Interface for Q-Functions that describe the policy on an environment.

    A Q-Function is a function that maps a state and an action to a real value. This
    value is the expected sum of discounted returns that the agent will encounter by
    executing an action at a state, and following the policy on the environment
    thereafter.

    Parameters
    ----------
    dim_state: int
        dimension of state.
    dim_action: int
        dimension of action.
    num_states: int, optional
        number of discrete states (None if state is continuous).
    num_actions: int, optional
        number of discrete actions (None if action is continuous).

    Attributes
    ----------
    dim_state: int
        dimension of state.
    dim_action: int
        dimension of action.
    num_states: int
        number of discrete states (None if state is continuous).
    num_actions: int, optional
        number of discrete actions (None if action is continuous).

    Methods
    -------
    __call__(state, action=None): float
        return the q-value of a given state.
    value(state, policy, num_samples=1): tensor
        return the value of that the function predicts after integrating the policy.
    discrete_state: bool
        Flag that indicates if state space is discrete.

    """

    def __init__(self, dim_state, dim_action, num_states=-1, num_actions=-1,
                 tau=1.):
        super().__init__(dim_state=dim_state, num_states=num_states, tau=tau)
        self.dim_action = dim_action
        self.num_actions = num_actions if num_actions is not None else -1
        self.discrete_action = self.num_actions >= 0
