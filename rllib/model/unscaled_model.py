"""Implementation of an Unscaled Model."""
from .abstract_model import AbstractModel
from rllib.dataset.datatypes import Observation
import gpytorch
import torch


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
        # Transform state and action.
        self.base_model.eval()

        for transformation in self.transformations:
            state, action, *_ = transformation(Observation(state, action))

        # Predict next-state
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            next_state = self.base_model(state, action)

        # Back-transform
        for transformation in reversed(self.transformations):
            state, action, reward, next_state, *_ = transformation.inverse(
                Observation(state, action, 0, next_state))

        return next_state
