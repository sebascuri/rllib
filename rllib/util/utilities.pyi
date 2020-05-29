from typing import Callable, List, Tuple, Union

import numpy as np
import torch.__spec__ as torch_mod
from torch import Tensor

from rllib.dataset.datatypes import (Array, Distribution, Gaussian, Reward,
                                     TupleDistribution)


def get_backend(array: Array) -> Union[np, torch_mod]: ...


def mellow_max(values: Array, omega: float = 1.) -> Array: ...


def integrate(function: Callable, distribution: Distribution,
              num_samples: int = 1) -> Tensor: ...


def tensor_to_distribution(args: TupleDistribution, **kwargs) -> Distribution: ...


def separated_kl(p: Gaussian, q: Gaussian) -> Tuple[Tensor, Tensor]: ...


def sample_mean_and_cov(sample: Tensor, diag: bool = False) -> Tuple[
    Tensor, Tensor]: ...


def safe_cholesky(covariance_matrix: Tensor, jitter: float = 1e-6) -> Tensor: ...


def moving_average_filter(x: Array, y: Array, horizon: int) -> Array: ...


class RewardTransformer(object):
    offset: float
    low: float
    high: float
    scale: float

    def __init__(self, offset: float = 0, low: float = -np.inf, high: float = np.inf,
                 scale: float = 1.) -> None: ...

    def __call__(self, reward: Reward) -> Reward: ...
