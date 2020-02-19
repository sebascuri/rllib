from rllib.dataset.datatypes import Observation, Distribution
from typing import List, Union, Callable
from numpy import ndarray
from torch import Tensor

Array = Union[Tensor, ndarray]

def _mc_value_slow(trajectory: List[Observation], gamma: float = 1.0) -> ndarray: ...

def mc_value(trajectory: List[Observation], gamma: float = 1.0) -> ndarray: ...

def sum_discounted_rewards(trajectory: List[Observation], gamma: float = 1.0
                           ) -> float: ...

def mellow_max(values: Array, omega: float = 1.) -> Array: ...


def integrate(function: Callable, distribution: Distribution, num_samples: int = 1) -> Tensor: ...
