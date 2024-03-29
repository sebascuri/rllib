from typing import Any, Optional, Sequence, Tuple, Type, TypeVar

import torch.nn
from torch import Tensor

from rllib.util.input_transformations import AbstractTransform

from .abstract_value_function import AbstractQFunction, AbstractValueFunction

T = TypeVar("T", bound="AbstractQFunction")

class NNValueFunction(AbstractValueFunction):
    input_transform: Optional[AbstractTransform]
    dimension: int
    nn: torch.nn.Module
    def __init__(
        self,
        layers: Sequence[int] = ...,
        biased_head: bool = ...,
        non_linearity: str = ...,
        input_transform: Optional[AbstractTransform] = ...,
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
        dim_state: Tuple[int],
        num_states: int = ...,
        tau: float = ...,
        input_transform: Optional[AbstractTransform] = ...,
        dim_reward: Tuple[int] = ...,
    ) -> T: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> Tensor: ...
    def embeddings(self, state: Tensor) -> Tensor: ...

class NNQFunction(AbstractQFunction):
    input_transform: Optional[AbstractTransform]
    nn: torch.nn.Module
    tau: float
    def __init__(
        self,
        layers: Sequence[int] = ...,
        biased_head: bool = ...,
        non_linearity: str = ...,
        jit_compile: bool = ...,
        input_transform: Optional[AbstractTransform] = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    @classmethod
    def from_other(cls: Type[T], other: T, copy: bool = ...) -> T: ...
    @classmethod
    def from_nn(
        cls: Type[T],
        module: torch.nn.Module,
        dim_state: Tuple[int],
        dim_action: Tuple[int],
        num_states: int = ...,
        num_actions: int = ...,
        tau: float = ...,
        input_transform: Optional[AbstractTransform] = ...,
        dim_reward: Tuple[int] = ...,
    ) -> T: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> Tensor: ...

class DuelingQFunction(NNQFunction):
    def __init__(
        self, average_or_max: str = ..., *args: Any, **kwargs: Any,
    ) -> None: ...
