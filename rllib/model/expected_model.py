"""Implementation Derived Models."""
import torch

from .transformed_model import TransformedModel


class ExpectedModel(TransformedModel):
    """Expected Model returns a Delta at the expected next state."""

    def forward(self, state, action):
        """Get Expected Next state."""
        ns, ns_scale_tril = self.next_state(state, action)

        return ns, torch.zeros_like(ns_scale_tril)  # , ns_scale_tril
