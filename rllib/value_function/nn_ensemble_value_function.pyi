from typing import List, Type, TypeVar

import torch.nn
from torch import Tensor

from .abstract_value_function import AbstractQFunction
from .nn_value_function import NNQFunction, NNValueFunction

T = TypeVar('T', bound='AbstractQFunction')


class NNEnsembleValueFunction(NNValueFunction):
    nn: torch.nn.ModuleList

    def __init__(self, dim_state: int,  num_heads: int, num_states: int = -1,
                 layers: List[int] = None,
                 biased_head: bool=True, non_linearity: str = 'ReLU', tau: float = 0.0,
                 input_transform: torch.nn.Module = None) -> None: ...

    @classmethod
    def from_value_function(cls: Type[T], value_function: NNValueFunction, num_heads: int) -> T: ...


    def forward(self, *args:Tensor, **kwargs) -> Tensor: ...


class NNEnsembleQFunction(NNQFunction):
    nn: torch.nn.ModuleList
    num_heads: int

    def __init__(self, dim_state: int, dim_action: int, num_heads: int,
                 num_states: int = -1, num_actions: int = -1,
                 layers: List[int] = None, biased_head: bool=True,
                 non_linearity: str = 'ReLU',  tau: float = 0.0,
                 input_transform: torch.nn.Module = None
                 ) -> None: ...

    @classmethod
    def from_q_function(cls: Type[T], q_function: NNQFunction, num_heads: int) -> T: ...


    def forward(self, *args: Tensor, **kwargs) -> Tensor: ...
