"""Implementation of Gaussian Processes State-Space Models."""
import gpytorch
import torch
import torch.nn

from rllib.util.gaussian_processes.exact_gp import ExactGP  # , MultitaskExactGP
from .abstract_model import AbstractModel


class ExactGPModel(AbstractModel):
    """An Exact GP State Space Model."""

    def __init__(self, states, actions, next_states,
                 likelihood=None, mean=None, kernel=None):
        dim_state = states.shape[-1]
        dim_action = actions.shape[-1]
        super().__init__(dim_state, dim_action)
        state_action = torch.cat((states, actions), dim=-1)
        # if dim_state == 1:
        likelihoods = tuple(gpytorch.likelihoods.GaussianLikelihood()
                            for _ in range(dim_state))
        gps = tuple(ExactGP(state_action, next_states[..., i:(i + 1)].transpose(-1, -2),
                            likelihoods[i],
                            mean, kernel) for i in range(dim_state))

        self.likelihood = torch.nn.ModuleList(likelihoods)
        self.gp = torch.nn.ModuleList(gps)

    def forward(self, state, action):
        """Get next state distribution."""
        state_action = torch.cat((state, action), dim=-1)
        if state_action.dim() < 2:
            state_action = state_action.unsqueeze(0)

        # for gp, likelihood in zip(self.gp, self.likelihood):
        out = [likelihood(gp(state_action))
               for gp, likelihood in zip(self.gp, self.likelihood)]
        # out = self.likelihood(*self.gp()))
        mean = torch.stack(tuple(o.mean for o in out), dim=0)
        cov = torch.stack(tuple(o.covariance_matrix for o in out), dim=0)

        return mean, cov
