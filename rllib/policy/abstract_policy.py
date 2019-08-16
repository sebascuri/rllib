"""Interface for policies."""


from abc import ABC, abstractmethod
import torch
from torch.distributions import MultivariateNormal, Categorical


class AbstractPolicy(ABC):
    """Interface for policies to control an environment.

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
    temperature: float, optional
        temperature scaling of output distribution.

    Attributes
    ----------
    dim_state: int
        dimension of state.
    dim_action: int
        dimension of action.
    num_states: int
        number of discrete states (None if state is continuous).
    num_actions: int
        number of discrete actions (None if action is continuous).

    Methods
    -------
    __call__(state): torch.distribution.Distribution
        return the action distribution that the policy suggests.
    parameters: generator
        return the policy parametrization.
    random: torch.distribution.Distribution
        return a uniform action distribution (same family as policy).
    discrete_state: bool
        Flag that indicates if state space is discrete.
    discrete_action: bool
        Flag that indicates if action space is discrete.
    """

    def __init__(self, dim_state, dim_action, num_states=None, num_actions=None,
                 temperature=1.):
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.num_states = num_states
        self.num_actions = num_actions
        self.temperature = temperature

    @abstractmethod
    def __call__(self, state):
        """Return the action distribution of the policy.

        Parameters
        ----------
        state: tensor

        Returns
        -------
        action: torch.distributions.Distribution

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

    def random(self, batch_size=None):
        """Return a uniform random distribution of the output space.

        Parameters
        ----------
        batch_size: int, optional
        Returns
        -------
        distribution: torch.distribution.Distribution
        """
        if self.discrete_action:  # Categorical
            distribution = Categorical(torch.ones(self.num_actions))
        else:
            distribution = MultivariateNormal(
                loc=torch.zeros(self.dim_action),
                covariance_matrix=self.temperature * torch.eye(self.dim_action))

        if batch_size is not None:
            return distribution.expand(batch_shape=(batch_size,))
        else:
            return distribution

    @property
    def discrete_state(self):
        """Flag that indicates if states are discrete.

        Returns
        -------
        bool
        """
        return self.num_states is not None

    @property
    def discrete_action(self):
        """Flag that indicates if actions are discrete.

        Returns
        -------
        bool
        """
        return self.num_actions is not None
