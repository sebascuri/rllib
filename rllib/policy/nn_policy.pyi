from typing import Any, List, Optional, Tuple, Type, TypeVar

import torch
from torch import Tensor

from rllib.dataset.datatypes import Action, TupleDistribution

from .abstract_policy import AbstractPolicy

T = TypeVar("T", bound="NNPolicy")

class NNPolicy(AbstractPolicy):
    input_transform: torch.nn.Module
    nn: torch.nn.Module
    def __init__(
        self,
        layers: Optional[List[int]] = ...,
        biased_head: bool = ...,
        non_linearity: str = ...,
        squashed_output: bool = ...,
        initial_scale: float = ...,
        input_transform: Optional[torch.nn.Module] = ...,
        *args: Any,
        **kwargs: Any
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
        deterministic: bool = ...,
        action_scale: Action = ...,
        goal: Optional[Tensor] = ...,
        input_transform: Optional[torch.nn.Module] = ...,
    ) -> T: ...
    def _preprocess_input_dim(self) -> Tuple: ...
    def _preprocess_state(self, state: Tensor) -> Tensor: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> TupleDistribution: ...
    def embeddings(self, state: Tensor, action: Optional[Tensor] = ...) -> Tensor: ...

class FelixPolicy(AbstractPolicy):
    def __init__(
        self,
        dim_state: Tuple,
        dim_action: Tuple,
        tau: float = ...,
        initial_scale: float = ...,
        deterministic: bool = ...,
        action_scale: Action = ...,
        goal: Optional[Tensor] = ...,
        input_transform: Optional[torch.nn.Module] = ...,
    ) -> None: ...
