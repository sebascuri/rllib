from typing import Any, Optional, Sequence, Tuple, Type, TypeVar

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
        layers: Sequence[int] = ...,
        biased_head: bool = ...,
        non_linearity: str = ...,
        input_transform: Optional[torch.nn.Module] = ...,
        jit_compile: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    @classmethod
    def from_other(cls: Type[T], other: T, copy: bool = ...) -> T: ...
    @classmethod
    def from_nn(
        cls: Type[T],
        module: torch.nn.Module,
        dim_state: Tuple,
        num_states: int = ...,
        tau: float = ...,
        input_transform: Optional[torch.nn.Module] = ...,
    ) -> T: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> Tensor: ...
    def embeddings(self, state: Tensor) -> Tensor: ...

class NNQFunction(AbstractQFunction):
    input_transform: torch.nn.Module
    nn: torch.nn.Module
    tau: float
    def __init__(
        self,
        dim_state: Tuple,
        dim_action: Tuple,
        num_states: int = ...,
        num_actions: int = ...,
        layers: Sequence[int] = ...,
        biased_head: bool = ...,
        non_linearity: str = ...,
        jit_compile: bool = ...,
        tau: float = ...,
        input_transform: Optional[torch.nn.Module] = ...,
    ) -> None: ...
    @classmethod
    def from_other(cls: Type[T], other: T, copy: bool = ...) -> T: ...
    @classmethod
    def from_nn(
        cls: Type[T],
        module: torch.nn.Module,
        dim_state: Tuple,
        dim_action: Tuple,
        num_states: int = ...,
        num_actions: int = ...,
        tau: float = ...,
        input_transform: Optional[torch.nn.Module] = ...,
    ) -> T: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> Tensor: ...

class DuelingQFunction(NNQFunction):
    def __init__(
        self, average_or_max: str = ..., *args: Any, **kwargs: Any,
    ) -> None: ...
