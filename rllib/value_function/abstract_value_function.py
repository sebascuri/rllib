"""Interface for value and q-functions."""

from abc import ABCMeta

import torch.nn as nn

from rllib.util.neural_networks import update_parameters
from rllib.util.utilities import integrate

__all__ = ['AbstractValueFunction', 'AbstractQFunction']


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

    def __init__(self, dim_state, num_states=None, tau=1.0):
        super().__init__()
        self.dim_state = dim_state
        self.num_states = num_states
        self.tau = tau
        self.discrete_state = self.num_states is not None

    def update_parameters(self, new_parameters):
        """Update policy parameters."""
        update_parameters(self.parameters(), new_parameters, tau=self.tau)


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

    def __init__(self, dim_state, dim_action, num_states=None, num_actions=None,
                 tau=1.):
        super().__init__(dim_state=dim_state, num_states=num_states, tau=tau)
        self.dim_action = dim_action
        self.num_actions = num_actions
        self.discrete_action = self.num_actions is not None

    def value(self, state, policy, num_samples=1):
        """Return the value of the state given a policy.

        Integrate Q(s, a) by sampling a from the policy.

        Parameters
        ----------
        state: tensor
        policy: AbstractPolicy
        num_samples: int, optional.
            number of samples when closed-form integration is not possible.

        Returns
        -------
        tensor
        """
        return integrate(lambda action: self(state, action), policy(state), num_samples)
