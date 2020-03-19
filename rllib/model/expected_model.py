"""Optimistic Model Implementation."""

import gpytorch
import torch

from .abstract_model import AbstractModel


class ExpectedModel(AbstractModel):
    r"""Model that predicts the next_state distribution.

    Given a model and a set of actions = actions,
    Return a Delta distribution at location:

    .. math:: \mu(state, action)

    Parameters
    ----------
    base_model: Model that returns a mean and stddev.
    """

    def __init__(self, base_model):
        super().__init__(dim_state=base_model.dim_state,
                         dim_action=base_model.dim_action,
                         num_states=base_model.num_states,
                         num_actions=base_model.num_actions)
        self.base_model = base_model

    def forward(self, states, actions):
        """Get next state distribution."""
        self.base_model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            prediction = self.base_model(states, actions)

        return prediction.mean, torch.zeros(1)
