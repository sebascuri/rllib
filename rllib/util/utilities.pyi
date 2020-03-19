from typing import Callable, Tuple, List

from torch import Tensor

from rllib.dataset.datatypes import Observation, Distribution, Array, Gaussian, \
    TupleDistribution
from rllib.value_function import AbstractValueFunction


def mellow_max(values: Array, omega: float = 1.) -> Array: ...


def integrate(function: Callable, distribution: Distribution,
              num_samples: int = 1) -> Tensor: ...


def discount_cumsum(returns: Array, gamma: float = 1.0) -> Array: ...


def discount_sum(returns: Array, gamma: float = 1.0) -> Array: ...


def mc_return(trajectory: List[Observation], gamma: float = 1.0,
              value_function: AbstractValueFunction = None): ...


def moving_average_filter(x: Array, y: Array, horizon: int) -> Array: ...


def tensor_to_distribution(args: TupleDistribution) -> Distribution: ...


def separated_kl(p: Gaussian, q: Gaussian) -> Tuple[Tensor, Tensor]: ...
