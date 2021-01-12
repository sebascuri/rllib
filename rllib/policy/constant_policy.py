"""Python Script Template."""
import torch

from rllib.policy import AbstractPolicy
from rllib.util.neural_networks.utilities import get_batch_size


class ConstantPolicy(AbstractPolicy):
    """A constant policy is independent of the state."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.discrete_action:
            self.logits = torch.rand(self.num_actions)
            self.logits.requires_grad = True
        else:
            self.mean = torch.randn(self.dim_action)
            self.std = 0.3 * torch.ones(self.dim_action)
            self.mean.requires_grad = True
            self.std.requires_grad = True

    def forward(self, state):
        """Compute action distribution."""
        batch_size = get_batch_size(state, self.dim_state)

        if self.discrete_action:
            logits = self.logits.repeat(*batch_size, 1)
            return logits
        else:
            mean = self.mean.repeat(*batch_size, 1)
            std = self.std.repeat(*batch_size, 1)
            return mean, std.diag_embed()
