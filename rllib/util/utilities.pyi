from rllib.dataset.datatypes import Observation, Distribution, Array
from typing import List, Callable
from numpy import ndarray
from torch import Tensor


# def _mc_value_slow(trajectory: List[Observation], gamma: float = 1.0) -> ndarray: ...
#
# def mc_value(trajectory: List[Observation], gamma: float = 1.0) -> ndarray: ...
#
# def sum_discounted_rewards(trajectory: List[Observation], gamma: float = 1.0
#                            ) -> float: ...

def mellow_max(values: Array, omega: float = 1.) -> Array: ...


def integrate(function: Callable, distribution: Distribution, num_samples: int = 1) -> Tensor: ...


def discount_cumsum(returns: Array, gamma: float = 1.0) -> Array: ...