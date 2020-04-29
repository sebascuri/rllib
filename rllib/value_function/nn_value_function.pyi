from typing import List, Type, TypeVar

import torch.nn
from torch import Tensor

from .abstract_value_function import AbstractValueFunction, AbstractQFunction

T = TypeVar('T', bound='AbstractQFunction')


class NNValueFunction(AbstractValueFunction):
    input_transform: torch.nn.Module
    dimension: int
    nn: torch.nn.Module

    def __init__(self, dim_state: int, num_states: int = -1, layers: List[int] = None,
                 biased_head: bool=True, non_linearity: str = 'ReLU', tau: float = 0.0,
                 input_transform: torch.nn.Module = None) -> None: ...

    @classmethod
    def from_other(cls: Type[T], other: T, copy: bool = True) -> T: ...

    @classmethod
    def from_nn(cls: Type[T], module: torch.nn.Module, dim_state: int, num_states: int=-1,
                tau: float = 0.0, input_transform: torch.nn.Module = None) -> T: ...

    def forward(self, *args: Tensor, **kwargs) -> Tensor: ...

    def embeddings(self, state: Tensor) -> Tensor: ...

class NNQFunction(AbstractQFunction):
    input_transform: torch.nn.Module
    nn: torch.nn.Module
    tau: float

    def __init__(self, dim_state: int, dim_action: int,
                 num_states: int = -1, num_actions: int = -1,
                 layers: List[int] = None, biased_head: bool=True,
                 non_linearity: str = 'ReLU',  tau: float = 0.0,
                 input_transform: torch.nn.Module = None
                 ) -> None: ...

    @classmethod
    def from_other(cls: Type[T], other: T, copy: bool = True) -> T: ...

    @classmethod
    def from_nn(cls: Type[T], module: torch.nn.Module, dim_state: int, dim_action: int,
                num_states: int = -1, num_actions: int = -1, tau: float = 0.0,
                input_transform: torch.nn.Module = None) -> T: ...


    def forward(self, *args: Tensor, **kwargs) -> Tensor: ...

