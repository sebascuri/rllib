from typing import Callable, Tuple, List,Union

import numpy as np
from torch import Tensor
import torch.__spec__ as torch_mod

from rllib.dataset.datatypes import Observation, Distribution, Array, Gaussian, \
    TupleDistribution
from rllib.value_function import AbstractValueFunction


def get_backend(array: Array) -> Union[np, torch_mod]: ...


def mellow_max(values: Array, omega: float = 1.) -> Array: ...


def integrate(function: Callable, distribution: Distribution,
              num_samples: int = 1) -> Tensor: ...


def discount_cumsum(returns: Array, gamma: float = 1.0) -> Array: ...


def discount_sum(returns: Tensor, gamma: float = 1.0) -> Array: ...


def mc_return(trajectory: List[Observation], gamma: float = 1.0,
              value_function: AbstractValueFunction = None, entropy_reg: float = 0.): ...


def tensor_to_distribution(args: TupleDistribution) -> Distribution: ...


def separated_kl(p: Gaussian, q: Gaussian) -> Tuple[Tensor, Tensor]: ...
