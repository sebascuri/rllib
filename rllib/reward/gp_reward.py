"""Interface for reward models."""

import gpytorch
import torch

from .abstract_reward import AbstractReward


class GPBanditReward(AbstractReward):
    """A Reward function that is defined through a GP."""

    def __init__(self, model):
        super().__init__()
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
