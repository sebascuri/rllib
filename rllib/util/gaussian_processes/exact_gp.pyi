"""Exact GP Model."""
import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import Likelihood
from gpytorch.means import Mean
from torch import Tensor


class ExactGP(gpytorch.models.ExactGP):
    """Exact GP Model."""

    def __init__(self, train_x: Tensor, train_y: Tensor, likelihood: Likelihood,
                 mean: Mean = None, kernel: Kernel = None) -> None: ...

    def forward(self, x: Tensor) -> MultivariateNormal: ...


class MultitaskExactGP(gpytorch.models.ExactGP):
    """Exact GP Model."""

    def __init__(self, train_x: Tensor, train_y: Tensor, likelihood: Likelihood,
                 num_tasks: int = 1, mean: Mean = None, kernel: Kernel = None) -> None: ...

    def forward(self, x: Tensor) -> MultivariateNormal: ...