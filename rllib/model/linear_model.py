"""Implementation of Linear Dynamical Models."""
from .abstract_model import AbstractModel
import torch
from gpytorch.distributions import Delta
from torch.distributions import MultivariateNormal


class LinearModel(AbstractModel):
    """A linear Gaussian state space model."""

    def __init__(self, a, b, noise: MultivariateNormal = None):
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a).float()
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b).float()

        super().__init__(*b.shape)

        self.a = a.t()
        self.b = b.t()
        self.noise = noise

    def __call__(self, state, action):
        """Get next state distribution."""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state).float()
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action).float()

        next_state = state @ self.a + action @ self.b
        if self.noise is None:
            return Delta(next_state)
        else:
            return MultivariateNormal(next_state + self.noise.mean,
                                      self.noise.covariance_matrix)
