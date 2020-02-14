"""Interface for value and q-functions."""


from abc import ABC, abstractmethod
import torch

__all__ = ['AbstractValueFunction', 'AbstractQFunction']


class AbstractValueFunction(ABC):
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

    def __init__(self, dim_state, num_states=None):
        self.dim_state = dim_state
        self.num_states = num_states

    @abstractmethod
    def __call__(self, state, action=None):
        """Return value function.

        Parameters
        ----------
        state: tensor
        action: tensor, optional

        Returns
        -------
        value: float
            value of state
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self):
        """Parameters that describe the policy.

        Returns
        -------
        generator
        """
        raise NotImplementedError

    @parameters.setter
    @abstractmethod
    def parameters(self, new_params):
        raise NotImplementedError

    @property
    def discrete_state(self):
        """Flag that indicates if states are discrete.

        Returns
        -------
        bool
        """
        return self.num_states is not None


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
        return the value of a given state.
    parameters: generator
        return the value function parametrization.
    discrete_state: bool
        Flag that indicates if state space is discrete.

    """

    def __init__(self, dim_state, dim_action, num_states=None, num_actions=None):
        super().__init__(dim_state=dim_state, num_states=num_states)
        self.dim_action = dim_action
        self.num_actions = num_actions

    @abstractmethod
    def __call__(self, state, action=None):
        """Return q-function value.

        If discrete_action and action is None, then it returns the q-value for all
        actions, else it returns the q-function for the specified state and action.

        Parameters
        ----------
        state: tensor
        action: tensor, optional

        Returns
        -------
        value: float or list
            value of state
        """
        raise NotImplementedError

    def value(self, state, policy, num_samples=0):
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
        pi = policy(state)
        if self.discrete_action:
            batch_size = state.shape[0]
            ans = torch.zeros(batch_size)
            for action in range(policy.num_actions):
                action = torch.tensor([action]).repeat(batch_size, 1)
                prob = pi.probs.gather(-1, action).squeeze()
                q_val = self(state, action.squeeze(-1))
                ans += prob * q_val

        else:
            ans = torch.zeros(state.shape[0])
            for _ in range(num_samples):
                q_val = self(state, pi.rsample())
                ans += q_val

        return ans

    @property
    def discrete_action(self):
        """Flag that indicates if actions are discrete.

        Returns
        -------
        bool
        """
        return self.num_actions is not None
