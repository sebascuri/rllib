"""Value and Q-Functions parametrized with Neural Networks."""

from .abstract_value_function import AbstractValueFunction, AbstractQFunction
from .nn_value_function import NNValueFunction, NNQFunction
from typing import List, Iterator
from torch import Tensor

class NNEnsembleValueFunction(AbstractValueFunction):
    dimension: int
    ensemble: List[NNValueFunction]

    def __init__(self, value_function: NNValueFunction = None,
                 dim_state: int = 1, num_states: int = None, layers: List[int] = None,
                 tau: float = 1.0, biased_head: bool=True, num_heads: int = 1) -> None: ...

    def __len__(self) -> int: ...

    def __getitem__(self, item: int) -> NNValueFunction: ...

    def __call__(self, state: Tensor, action: Tensor = None) -> List[Tensor]: ...

    @property
    def parameters(self) -> List[Iterator]: ...

    @parameters.setter
    def parameters(self, value: List[Iterator]) -> None: ...


    def embeddings(self, state: Tensor) -> Tensor: ...


class NNEnsembleQFunction(AbstractQFunction):
    ensemble: List[NNQFunction]

    def __init__(self, q_function: NNQFunction = None,
                 dim_state: int = 1, dim_action: int = 1,
                 num_states: int = None, num_actions: int = None,
                 layers: List[int] = None,  tau: float = 1.0, biased_head: bool=True,
                 num_heads: int = 1) -> None: ...

    def __len__(self) -> int: ...

    def __getitem__(self, item: int) -> NNValueFunction: ...

    def __call__(self, state: Tensor, action: Tensor = None) -> List[Tensor]: ...

    @property
    def parameters(self) -> List[Iterator]: ...

    @parameters.setter
    def parameters(self, new_params: List[Iterator]) -> None: ...
