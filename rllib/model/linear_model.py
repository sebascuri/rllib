"""Implementation of Linear Dynamical Models."""
import torch
from torch.distributions import MultivariateNormal

from rllib.util.neural_networks.utilities import to_torch

from .abstract_model import AbstractModel


class LinearModel(AbstractModel):
    """A linear Gaussian state space model."""

    def __init__(self, a, b, noise: MultivariateNormal = None, *args, **kwargs):
        a, b = to_torch(a), to_torch(b)

        super().__init__(
            dim_state=a.shape[1], dim_action=b.shape[1], deterministic=noise is None
        )

        self.a = a.t()
        self.b = b.t()
        self.noise = noise

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See AbstractModel.default()."""
        dim_state, dim_action = environment.dim_state[0], environment.dim_action[0]
        return super().default(
            environment,
            a=kwargs.pop("a", torch.zeros(dim_state, dim_state)),
            b=kwargs.pop("b", torch.zeros(dim_state, dim_action)) * args,
            **kwargs,
        )

    def forward(self, state, action, next_state=None):
        """Get next state distribution."""
        state, action = to_torch(state), to_torch(action)

        next_state = state @ self.a + action @ self.b
        if self.noise is None:
            return next_state, torch.zeros(1)
        else:
            return next_state + self.noise.mean, self.noise.covariance_matrix
