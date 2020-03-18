"""Optimistic Model Implementation."""

from .abstract_model import AbstractModel
import torch
import gpytorch


class OptimisticModel(AbstractModel):
    r"""Model that predicts the next_state distribution.

    Given a model and a set of actions = [actions, eta],
    Return a Delta distribution at location:

    .. math:: \mu(state, action) + \eta * \Sigma^(1/2)(state, action)

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
        """Predict the next state distribution."""
        control = actions[..., :-self.dim_states]
        optimism = actions[..., -self.dim_states:]
        self.base_model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            prediction = self.base_model(states, control)
        optimism = torch.clamp(optimism, -1., 1.)
        return prediction.mean + prediction.stddev * optimism, torch.zeros(1)
