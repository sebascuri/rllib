from typing import List, Optional, Type, TypeVar

import torch.nn
from torch import Tensor

from .abstract_value_function import AbstractQFunction, AbstractValueFunction

T = TypeVar("T", bound="AbstractQFunction")

class NNValueFunction(AbstractValueFunction):
    input_transform: torch.nn.Module
    dimension: int
    nn: torch.nn.Module
    def __init__(
        self,
        dim_state: int,
        num_states: int = -1,
        layers: Optional[List[int]] = ...,
        biased_head: bool = ...,
        non_linearity: str = ...,
        tau: float = ...,
        input_transform: Optional[torch.nn.Module] = ...,
    ) -> None: ...
    @classmethod
    def from_other(cls: Type[T], other: T, copy: bool = ...) -> T: ...
    @classmethod
    def from_nn(
        cls: Type[T],
        module: torch.nn.Module,
        dim_state: int,
        num_states: int = ...,
        tau: float = ...,
        input_transform: Optional[torch.nn.Module] = ...,
    ) -> T: ...
    def forward(self, *args: Tensor, **kwargs) -> Tensor: ...
    def embeddings(self, state: Tensor) -> Tensor: ...

class NNQFunction(AbstractQFunction):
    input_transform: torch.nn.Module
    nn: torch.nn.Module
    tau: float
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        num_states: int = ...,
        num_actions: int = ...,
        layers: Optional[List[int]] = ...,
        biased_head: bool = ...,
        non_linearity: str = ...,
        tau: float = ...,
        input_transform: Optional[torch.nn.Module] = ...,
    ) -> None: ...
    @classmethod
    def from_other(cls: Type[T], other: T, copy: bool = ...) -> T: ...
    @classmethod
    def from_nn(
        cls: Type[T],
        module: torch.nn.Module,
        dim_state: int,
        dim_action: int,
        num_states: int = ...,
        num_actions: int = ...,
        tau: float = ...,
        input_transform: Optional[torch.nn.Module] = ...,
    ) -> T: ...
    def forward(self, *args: Tensor, **kwargs) -> Tensor: ...
