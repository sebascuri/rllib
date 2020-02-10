from .abstract_value_function import AbstractValueFunction, AbstractQFunction
from torch import Tensor
from typing import List, Iterator
from rllib.util.neural_networks import DeterministicNN
from rllib.policy import AbstractPolicy


class NNValueFunction(AbstractValueFunction):
    dimension: int
    value_function: DeterministicNN
    _tau: float

    def __init__(self, dim_state: int, num_states: int = None, layers: List[int] = None,
                 tau: float = 1.0, biased_head: bool=True) -> None: ...
    def __call__(self, state: Tensor, action: Tensor = None) -> Tensor: ...

    @property
    def parameters(self) -> Iterator: ...

    @parameters.setter
    def parameters(self, value: Iterator) -> None: ...


    def embeddings(self, state: Tensor) -> Tensor: ...

class NNQFunction(AbstractQFunction):
    q_function: DeterministicNN
    _tau: float

    def __init__(self, dim_state: int, dim_action: int,
                 num_states: int = None, num_actions: int = None,
                 layers: List[int] = None,  tau: float = 1.0, biased_head: bool=True
                 ) -> None: ...

    def __call__(self, state: Tensor, action: Tensor = None) -> Tensor: ...

    @property
    def parameters(self) -> Iterator: ...

    @parameters.setter
    def parameters(self, new_params: Iterator) -> None: ...
