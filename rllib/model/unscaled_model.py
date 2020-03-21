"""Implementation of an Unscaled Model."""
# import gpytorch
import torch

from rllib.dataset.datatypes import RawObservation
from .abstract_model import AbstractModel


class UnscaledModel(AbstractModel):
    """Unscaled Model computes the next state distribution."""

    def __init__(self, base_model, transformations):
        super().__init__(dim_state=base_model.dim_state,
                         dim_action=base_model.dim_action,
                         num_states=base_model.num_states,
                         num_actions=base_model.num_actions)
        self.base_model = base_model
        self.transformations = transformations

    def forward(self, state, action):
        """Predict next state distribution."""
        batch_size = state.shape[0:-1]
        for transformation in self.transformations:
            state, action, *_ = transformation(RawObservation(state, action))

        # Predict next-state
        # with torch.no_grad(), gpytorch.settings.fast_pred_var():
        next_state = self.base_model(state, action)
        if next_state[0].shape[-1] is not self.dim_state:
            mean = next_state[0].transpose(0, -1)
            idx = torch.arange(0, next_state[0].shape[-1])
            var = next_state[1][..., idx, idx].transpose(0, -1)

            mean = mean.reshape(*batch_size, self.dim_state)
            var = var.reshape(*batch_size, self.dim_state)
            cov = torch.diag_embed(var)
            next_state = mean, cov

        # Back-transform
        for transformation in reversed(self.transformations):
            state, action, reward, next_state, *_ = transformation.inverse(
                RawObservation(state, action, 0, next_state))

        return next_state
