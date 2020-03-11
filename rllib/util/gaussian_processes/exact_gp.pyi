"""Exact GP Model."""
import gpytorch
from torch import Tensor
from gpytorch.likelihoods import Likelihood
from gpytorch.means import Mean
from gpytorch.kernels import Kernel
from gpytorch.distributions import MultivariateNormal

class ExactGP(gpytorch.models.ExactGP):
    """Exact GP Model."""

    def __init__(self, train_x: Tensor, train_y: Tensor, likelihood: Likelihood,
                 mean: Mean = None, kernel: Kernel = None) -> None: ...

    def forward(self, x: Tensor) -> MultivariateNormal: ...