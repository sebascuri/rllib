"""Interface for value and q-functions."""

from abc import ABCMeta

import torch.nn as nn


class AbstractQFunction(nn.Module, metaclass=ABCMeta):
    """Interface for Q-Functions.

    A Q-Function is a function that maps a state and an action to a real value. This
    value is the expected sum of discounted returns that the agent will encounter by
    executing an action at a state, and following the policy on the environment
    thereafter.

    Parameters
    ----------
    dim_state: Tuple.
        Dimension of the state.
    dim_action: Tuple.
        Dimension of the action
    num_states: int.
        Number of states in discrete environments.
    num_actions: int.
        Number of actions in discrete environments.
    tau: float.
        Low-pass filter parameter to update the value function.

    """

    def __init__(
        self,
        dim_state,
        dim_action,
        num_states=-1,
        num_actions=-1,
        tau=0.0,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim_action = dim_action
        self.num_actions = num_actions if num_actions is not None else -1
        self.discrete_action = self.num_actions >= 0

        self.dim_state = dim_state
        self.num_states = num_states if num_states is not None else -1
        self.tau = tau
        self.discrete_state = self.num_states >= 0


class AbstractValueFunction(AbstractQFunction):
    """Interface for Value-Functions.

    A Value-Function is a function that maps a state to a real value. This value is the
    expected sum of discounted returns that the agent will encounter by following the
    policy on the environment.

    Parameters
    ----------
    dim_state: Tuple.
        Dimension of the state.
    num_states: int.
        Number of states in discrete environments.
    tau: float.
        Low-pass filter parameter to update the value function.
    """

    def __init__(self, dim_state, num_states=-1, tau=0.0):
        super().__init__(dim_state, dim_action=0, num_states=num_states, tau=tau)
