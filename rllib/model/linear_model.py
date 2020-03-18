"""Implementation of Linear Dynamical Models."""
from .abstract_model import AbstractModel
import torch
from torch.distributions import MultivariateNormal


class LinearModel(AbstractModel):
    """A linear Gaussian state space model."""

    def __init__(self, a, b, noise: MultivariateNormal = None):
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=torch.get_default_dtype())
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, dtype=torch.get_default_dtype())

        super().__init__(dim_state=b.shape[0], dim_action=b.shape[1])

        self.a = a.t()
        self.b = b.t()
        self.noise = noise

    def forward(self, state, action):
        """Get next state distribution."""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.get_default_dtype())
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.get_default_dtype())

        next_state = state @ self.a + action @ self.b
        if self.noise is None:
            return next_state, torch.zeros(1)
        else:
            return next_state + self.noise.mean, self.noise.covariance_matrix
