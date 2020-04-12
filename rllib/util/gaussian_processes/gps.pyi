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
                 mean: Mean = None, kernel: Kernel = None,
                 jitter: float = 1e-6) -> None: ...

    @property
    def output_scale(self) -> float: ...

    @output_scale.setter
    def output_scale(self, new_output_scale: float) -> None: ...

    @property
    def length_scale(self) -> float: ...

    @length_scale.setter
    def length_scale(self, new_length_scale: float) -> None: ...

    def forward(self, x: Tensor) -> MultivariateNormal: ...


class SparseGP(ExactGP):
    def __init__(self, train_x: Tensor, train_y: Tensor, likelihood: Likelihood,
                 inducing_points: Tensor, mean: Mean = None, kernel: Kernel = None,
                 approximation: str = 'DTC') -> None: ...

    @property
    def output_scale(self) -> float: ...

    @output_scale.setter
    def output_scale(self, new_output_scale: float) -> None: ...

    @property
    def length_scale(self) -> float: ...

    @length_scale.setter
    def length_scale(self, new_length_scale: float) -> None: ...

    def forward(self, x: Tensor) -> MultivariateNormal: ...


class RandomFeatureGP(ExactGP):
    def __init__(self, train_x: Tensor, train_y: Tensor, likelihood: Likelihood,
                 num_features: int, mean: Mean = None, kernel: Kernel = None,
                 outputscale: float = 1., lengthscale: float = 1.) -> None: ...

    @property
    def output_scale(self) -> float: ...

    @output_scale.setter
    def output_scale(self, new_output_scale: float) -> None: ...

    @property
    def length_scale(self) -> float: ...

    @length_scale.setter
    def length_scale(self, new_length_scale: float) -> None: ...

    def forward(self, x: Tensor) -> MultivariateNormal: ...
