"""Interface for reward models."""

from .abstract_reward import AbstractReward
import torch
import gpytorch


class GPBanditReward(AbstractReward):
    """A Reward function that is defined through a GP."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, state, action):
        """Compute the reward for a given state, action pairs."""
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if type(action) is not torch.Tensor:
                action = torch.tensor(action)
            if action.ndim == 0:
                action = action.unsqueeze(0)

            pred = self.model(action).mean
            return self.model.likelihood(pred)
