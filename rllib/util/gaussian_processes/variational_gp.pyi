"""Exact GP Model."""
import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.means import Mean
from torch import Tensor

class ApproximateGPModel(gpytorch.models.ApproximateGP):
    """Exact GP Model."""

    def __init__(
        self,
        inducing_points: Tensor,
        learn_loc: bool = True,
        mean: Mean = None,
        kernel: Kernel = None,
    ) -> None: ...
    def forward(self, x: Tensor) -> MultivariateNormal: ...
