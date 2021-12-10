"""Empty model that works as a placeholder."""

import torch

from .abstract_model import AbstractModel


class EmptyModel(AbstractModel):
    """Empty model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state, action, next_state=None):
        """Compute output of model."""
        if self.model_kind == "rewards":
            return torch.zeros(1), torch.zeros(1)
        elif self.model_kind == "dynamics":
            if self.discrete_state:
                return torch.zeros(1)
            else:
                return torch.zeros(1), torch.zeros(1)
        elif self.model_kind == "termination":
            return torch.zeros(1)
        return torch.zeros(1)
