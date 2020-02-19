from torch import Tensor
from abc import abstractmethod
from ..abstract_policy import AbstractPolicy
from rllib.value_function import AbstractQFunction
from rllib.util import ParameterDecay
from rllib.dataset.datatypes import Distribution
from typing import Iterator


class AbstractQFunctionPolicy(AbstractPolicy):
    q_function: AbstractQFunction
    param: ParameterDecay

    def __init__(self, q_function: AbstractQFunction, start: float, end: float = None,
                 decay: float = None) -> None: ...

    @abstractmethod
    def __call__(self, state: Tensor) -> Distribution: ...

    @property
    def parameters(self) -> Iterator: ...

    @parameters.setter
    def parameters(self, new_params: Iterator) -> None: ...

