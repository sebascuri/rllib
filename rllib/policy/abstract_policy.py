"""Interface for policies."""

from abc import ABCMeta
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
from rllib.util.neural_networks import update_parameters


class AbstractPolicy(nn.Module, metaclass=ABCMeta):
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
    tau: float, optional.
        low pass coefficient for parameter update.
    deterministic: bool, optional.
        flag that indicates if the policy is deterministic.

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
                 tau=1.0, deterministic=False):
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.num_states = num_states
        self.num_actions = num_actions
        self.deterministic = deterministic
        self.tau = tau

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
                covariance_matrix=torch.eye(self.dim_action))

        if batch_size is not None:
            return distribution.expand(batch_shape=torch.Size((batch_size,)))
        else:
            return distribution

    def update(self, observation):
        """Update policy parameters."""
        pass

    def update_parameters(self, new_parameters):
        """Update policy parameters."""
        update_parameters(self.parameters(), new_parameters, tau=self.tau)

    @property
    def discrete_state(self):
        """Flag that indicates if states are discrete.

        Returns
        -------
        flag: bool
            whether the state space is discrete.

        """
        return self.num_states is not None

    @property
    def discrete_action(self):
        """Flag that indicates if actions are discrete.

        Returns
        -------
        flag: bool
            whether the actions space is discrete.

        """
        return self.num_actions is not None
