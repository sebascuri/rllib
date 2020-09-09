"""Implementation Derived Models."""
import torch

from .transformed_model import TransformedModel


class ExpectedModel(TransformedModel):
    """Expected Model returns a Delta at the expected next state."""

    def __init__(self, *args, **kwargs):
        super().__init__(deterministic=True, *args, **kwargs)

    def forward(self, state, action, next_state=None):
        """Get Expected Next state."""
        prediction_tuple = self.predict(state, action[..., : self.dim_action[0]])
        if len(prediction_tuple) == 1:
            return prediction_tuple
        else:
            mean, covariance = prediction_tuple
            return mean, torch.zeros_like(covariance)
