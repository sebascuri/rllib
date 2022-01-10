"""Zero policy implementation."""

import torch

from rllib.util.neural_networks.utilities import get_batch_size

from .abstract_policy import AbstractPolicy


class ZeroPolicy(AbstractPolicy):
    """Zero Policy implementation of AbstractPolicy base class.

    This policy will always return a zero action.
    It is only implemented for continuous action environments.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.discrete_action:
            raise NotImplementedError("Actions can't be discrete.")

    def forward(self, state):
        """Get distribution over actions."""
        batch_size = get_batch_size(state, self.dim_state)
        if batch_size:
            return torch.zeros((batch_size,) + self.dim_action)
        else:
            return self.zeros(self.dim_action)
