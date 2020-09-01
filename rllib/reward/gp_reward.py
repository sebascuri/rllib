"""Interface for reward models."""

import gpytorch
import torch

from rllib.model import AbstractModel


class GPBanditReward(AbstractModel):
    """A Reward function that is defined through a GP."""

    def __init__(self, model):
        super().__init__(model_kind="rewards", dim_state=(), dim_action=())
        self.model = model

    def forward(self, state, action, next_state):
        """Compute the reward for a given state, action pairs."""
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if type(action) is not torch.Tensor:
                action = torch.tensor(action)
            if action.ndim == 0:
                action = action.unsqueeze(0)

            pred = self.model(action).mean
            out = self.model.likelihood(pred)

        if isinstance(out, gpytorch.distributions.MultivariateNormal):
            return out.mean, out.lazy_covariance_matrix
        else:
            return out.mean, out.variance.unsqueeze(0)
