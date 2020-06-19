"""Exact GP Model."""
from typing import Optional

import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.means import Mean
from torch import Tensor

class ApproximateGPModel(gpytorch.models.ApproximateGP):
    """Exact GP Model."""

    mean_module: Mean
    covar_module: Kernel
    def __init__(
        self,
        inducing_points: Tensor,
        learn_loc: bool = ...,
        mean: Optional[Mean] = ...,
        kernel: Optional[Kernel] = ...,
    ) -> None: ...
    def forward(self, x: Tensor) -> MultivariateNormal: ...
