from typing import Any, List, Optional, Tuple, Type, TypeVar

import torch.nn
from torch import Tensor

from rllib.value_function import AbstractQFunction, NNQFunction, NNValueFunction

T = TypeVar("T", bound="AbstractQFunction")

class NNEnsembleValueFunction(NNValueFunction):
    nn: torch.nn.ModuleList
    def __init__(
        self,
        dim_state: Tuple,
        num_heads: int,
        num_states: int = ...,
        layers: Optional[List[int]] = ...,
        biased_head: bool = ...,
        non_linearity: str = ...,
        tau: float = ...,
        input_transform: Optional[torch.nn.Module] = ...,
    ) -> None: ...
    @classmethod
    def from_value_function(
        cls: Type[T], value_function: NNValueFunction, num_heads: int
    ) -> T: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> Tensor: ...

class NNEnsembleQFunction(NNQFunction):
    nn: torch.nn.ModuleList
    num_heads: int
    def __init__(
        self,
        dim_state: Tuple,
        dim_action: Tuple,
        num_heads: int,
        num_states: int = ...,
        num_actions: int = ...,
        layers: Optional[List[int]] = ...,
        biased_head: bool = ...,
        non_linearity: str = ...,
        tau: float = ...,
        input_transform: Optional[torch.nn.Module] = ...,
    ) -> None: ...
    @classmethod
    def from_q_function(cls: Type[T], q_function: NNQFunction, num_heads: int) -> T: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> Tensor: ...
