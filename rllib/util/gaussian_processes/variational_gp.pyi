"""Exact GP Model."""
import gpytorch
from torch import Tensor
from gpytorch.likelihoods import Likelihood
from gpytorch.means import Mean
from gpytorch.kernels import Kernel
from gpytorch.distributions import MultivariateNormal

class ApproximateGPModel(gpytorch.models.ApproximateGP):
    """Exact GP Model."""

    def __init__(self, inducing_points: Tensor, learn_loc: bool = True,
                 mean: Mean = None, kernel: Kernel = None) -> None: ...

    def forward(self, x: Tensor) -> MultivariateNormal: ...
