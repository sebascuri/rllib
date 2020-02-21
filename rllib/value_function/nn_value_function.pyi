from .abstract_value_function import AbstractValueFunction, AbstractQFunction
from torch import Tensor
from typing import List, Union
from rllib.util.neural_networks import DeterministicNN


class NNValueFunction(AbstractValueFunction):
    dimension: int
    nn: DeterministicNN

    def __init__(self, dim_state: int, num_states: int = None, layers: List[int] = None,
                 tau: float = 1.0, biased_head: bool=True) -> None: ...

    def forward(self, *args: Tensor, **kwargs) -> Tensor: ...

    def embeddings(self, state: Tensor) -> Tensor: ...

class NNQFunction(AbstractQFunction):
    nn: DeterministicNN
    tau: float

    def __init__(self, dim_state: int, dim_action: int,
                 num_states: int = None, num_actions: int = None,
                 layers: List[int] = None,  tau: float = 1.0, biased_head: bool=True
                 ) -> None: ...

    def forward(self, *args: Tensor, **kwargs) -> Tensor: ...



class TabularValueFunction(NNValueFunction):
    def __init__(self, num_states: int, tau: float = 1.0,
                 biased_head: bool = False) -> None: ...

    @property
    def table(self) -> Tensor: ...

    def set_value(self, state: Union[Tensor, int],
                  new_value: Union[Tensor, float]) -> None: ...


class TabularQFunction(NNQFunction):
    def __init__(self, num_states: int, num_actions: int, tau: float = 1.0,
                 biased_head: bool = False) -> None: ...

    def table(self) -> Tensor: ...

    def set_value(self, state: Union[Tensor, int],
                  action: Union[Tensor, int], new_value: Union[Tensor, float]
                  ) -> None: ...
