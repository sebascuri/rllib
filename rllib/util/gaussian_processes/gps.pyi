"""Exact GP Model."""
import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import Likelihood
from gpytorch.means import Mean
from torch import Tensor
from typing import Union


class ExactGP(gpytorch.models.ExactGP):
    """Exact GP Model."""

    def __init__(self, train_x: Tensor, train_y: Tensor, likelihood: Likelihood,
                 mean: Mean = None, kernel: Kernel = None,
                 jitter: float = 1e-6) -> None: ...

    @property
    def name(self) -> str: ...

    @property
    def output_scale(self) -> Tensor: ...

    @output_scale.setter
    def output_scale(self, new_output_scale: Union[float, Tensor]) -> None: ...

    @property
    def length_scale(self) -> Tensor: ...

    @length_scale.setter
    def length_scale(self, new_length_scale: Union[float, Tensor]) -> None: ...

    def forward(self, x: Tensor) -> MultivariateNormal: ...


class SparseGP(ExactGP):
    def __init__(self, train_x: Tensor, train_y: Tensor, likelihood: Likelihood,
                 inducing_points: Tensor, approximation: str = 'DTC',
                 mean: Mean = None, kernel: Kernel = None) -> None: ...

    def set_inducing_points(self, inducing_points: Tensor) -> None: ...

    def forward(self, x: Tensor) -> MultivariateNormal: ...

    def __call__(self, *args: Tensor, **kwargs) -> MultivariateNormal: ...


class RandomFeatureGP(ExactGP):
    def __init__(self, train_x: Tensor, train_y: Tensor, likelihood: Likelihood,
                 num_features: int, approximation='rff',
                 mean: Mean = None, kernel: Kernel = None) -> None: ...

    @ExactGP.length_scale.setter  # type: ignore
    def length_scale(self, new_length_scale: Union[float, Tensor]) -> None: ...

    def sample_features(self) -> Union[Tensor, Tensor, Tensor]: ...

    @property
    def num_features(self) -> int: ...

    @num_features.setter
    def num_features(self, value: int) -> None: ...

    @property
    def scale(self) -> Tensor: ...

    def forward(self, x: Tensor) -> Tensor: ...

    def __call__(self, *args: Tensor, **kwargs) -> MultivariateNormal: ...
