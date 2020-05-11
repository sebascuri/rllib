from typing import Callable, Tuple, List, Union

import numpy as np
from torch import Tensor
import torch.__spec__ as torch_mod

from rllib.dataset.datatypes import Distribution, Array, Gaussian, TupleDistribution


def get_backend(array: Array) -> Union[np, torch_mod]: ...


def mellow_max(values: Array, omega: float = 1.) -> Array: ...


def integrate(function: Callable, distribution: Distribution,
              num_samples: int = 1) -> Tensor: ...


def tensor_to_distribution(args: TupleDistribution) -> Distribution: ...


def separated_kl(p: Gaussian, q: Gaussian) -> Tuple[Tensor, Tensor]: ...


def sample_mean_and_cov(sample: Tensor) -> Tuple[Tensor, Tensor]: ...


def safe_cholesky(covariance_matrix: Tensor, jitter: float = 1e-6) -> Tensor: ...


def moving_average_filter(x: Array, y: Array, horizon: int) -> Array: ...
