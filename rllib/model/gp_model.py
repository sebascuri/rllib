"""Implementation of Gaussian Processes State-Space Models."""
import gpytorch
import torch

from rllib.util.gaussian_processes.exact_gp import ExactGP, MultitaskExactGP
from .abstract_model import AbstractModel


class ExactGPModel(AbstractModel):
    """An Exact GP State Space Model."""

    def __init__(self, states, actions, next_states,
                 likelihood=None, mean=None, kernel=None):
        dim_state = states.shape[-1]
        dim_action = actions.shape[-1]
        super().__init__(dim_state, dim_action)
        state_action = torch.cat((states, actions), dim=-1)
        if dim_state == 1:
            if likelihood is None:
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.likelihood = likelihood
            self.gp = ExactGP(state_action, next_states, likelihood, mean, kernel)
        else:
            if likelihood is None:
                likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                    num_tasks=dim_state)
            self.likelihood = likelihood
            self.gp = MultitaskExactGP(train_x=state_action, train_y=next_states,
                                       likelihood=likelihood,
                                       mean=mean, kernel=kernel, num_tasks=dim_state)

    def forward(self, state, action):
        """Get next state distribution."""
        state_action = torch.cat((state, action), dim=-1)
        if state_action.dim() < 2:
            state_action = state_action.unsqueeze(0)

        out = self.likelihood(self.gp(state_action))
        return out.mean, out.covariance_matrix
