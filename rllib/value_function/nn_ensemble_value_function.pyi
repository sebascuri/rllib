from typing import Any, Type, TypeVar

import torch.nn
from torch import Tensor

from rllib.value_function import AbstractQFunction, NNQFunction, NNValueFunction

T = TypeVar("T", bound="AbstractQFunction")

class NNEnsembleValueFunction(NNValueFunction):
    nn: torch.nn.ModuleList
    num_heads: int
    def __init__(self, num_heads: int, *args: Any, **kwargs: Any) -> None: ...
    @classmethod
    def from_value_function(
        cls: Type[T], value_function: NNValueFunction, num_heads: int
    ) -> T: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> Tensor: ...

class NNEnsembleQFunction(NNQFunction):
    nn: torch.nn.ModuleList
    num_heads: int
    def __init__(self, num_heads: int, *args: Any, **kwargs: Any) -> None: ...
    @classmethod
    def from_q_function(cls: Type[T], q_function: NNQFunction, num_heads: int) -> T: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> Tensor: ...
