"""Interface for policies."""

from abc import ABCMeta

import torch
import torch.jit
import torch.nn as nn


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
    forward(state): torch.distribution.Distribution
        return the action distribution that the policy suggests.
    random: torch.distribution.Distribution
        return a uniform action distribution (same family as policy).
    discrete_state: bool
        Flag that indicates if state space is discrete.
    discrete_action: bool
        Flag that indicates if action space is discrete.
    """

    def __init__(self, dim_state, dim_action, num_states=-1, num_actions=-1,
                 tau=0.0, deterministic=False):
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.num_states = num_states if num_states is not None else -1
        self.num_actions = num_actions if num_actions is not None else -1
        self.deterministic = deterministic
        self.discrete_state = self.num_states >= 0
        self.discrete_action = self.num_actions >= 0
        self.tau = tau

    def random(self, batch_size=None):
        """Return a uniform random distribution of the output space.

        Parameters
        ----------
        batch_size: tuple, optional

        Returns
        -------
        distribution: torch.distribution.Distribution

        """
        if self.discrete_action:  # Categorical
            # distribution = Categorical(torch.ones(self.num_actions))
            if batch_size is None:
                return torch.ones(self.num_actions)
            else:
                return torch.ones(*batch_size, self.num_actions)
        else:
            cov = torch.eye(self.dim_action)
            if batch_size is None:
                return torch.zeros(self.dim_action), cov
            else:
                return torch.zeros(*batch_size, self.dim_action), \
                       cov.expand(*batch_size, self.dim_action, self.dim_action)

    @torch.jit.export
    def reset(self):
        """Reset policy parameters (for example internal states)."""
        pass

    @torch.jit.export
    def update(self):
        """Update policy parameters."""
        pass
