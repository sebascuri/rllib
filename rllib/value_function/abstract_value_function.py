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
        tau=5e-3,
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

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """Get a default q-function for the environment."""
        return cls(
            dim_state=environment.dim_state,
            num_states=environment.num_states,
            dim_action=environment.dim_action,
            num_actions=environment.num_actions,
            *args,
            **kwargs,
        )


class AbstractValueFunction(AbstractQFunction):
    """Interface for Value-Functions.

    A Value-Function is a function that maps a state to a real value. This value is the
    expected sum of discounted returns that the agent will encounter by following the
    policy on the environment.
    """

    def __init__(self, *args, **kwargs):
        dim_action = kwargs.pop("dim_action", None)
        super().__init__(dim_action=(0,), *args, **kwargs)
        if dim_action is not None:
            kwargs.update(dim_action=dim_action)

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """Get a default value-function for the environment."""
        return super().default(environment, *args, **kwargs)
