"""Implementation of different Neural Networks with pytorch."""

from typing import Any, Dict, Optional, Sequence, Tuple, Type, TypeVar

import torch.nn as nn
from torch import Tensor

T = TypeVar("T", bound="FeedForwardNN")

class FeedForwardNN(nn.Module):
    kwargs: Dict
    embedding_dim: int
    hidden_layers: nn.Sequential
    head: nn.Module
    squashed_output: bool
    _init_scale_transformed: Tensor
    _min_scale: float
    _max_scale: float
    log_scale: bool
    def __init__(
        self,
        in_dim: Tuple,
        out_dim: Tuple,
        layers: Optional[Sequence[int]] = ...,
        non_linearity: str = ...,
        biased_head: bool = ...,
        squashed_output: bool = ...,
        initial_scale: float = ...,
        log_scale: bool = ...,
    ) -> None: ...
    @classmethod
    def from_other(cls: Type[T], other: T, copy: bool = ...) -> T: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> Any: ...
    def last_layer_embeddings(self, x: Tensor) -> Tensor: ...

class DeterministicNN(FeedForwardNN):
    def forward(self, *args: Tensor, **kwargs: Any) -> Tensor: ...

class CategoricalNN(FeedForwardNN): ...

class HeteroGaussianNN(FeedForwardNN):
    _scale: nn.Linear
    def forward(self, *args: Tensor, **kwargs: Any) -> Tuple[Tensor, Tensor]: ...

class HomoGaussianNN(FeedForwardNN):
    _scale: nn.Parameter
    def forward(self, *args: Tensor, **kwargs: Any) -> Tuple[Tensor, Tensor]: ...

class Ensemble(HeteroGaussianNN):
    num_heads: int
    head_ptr: int
    deterministic: bool
    prediction_strategy: str
    def __init__(
        self,
        in_dim: Tuple,
        out_dim: Tuple,
        num_heads: int,
        prediction_strategy: str = ...,
        deterministic: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    @classmethod
    def from_feedforward(
        cls: Type[T],
        other: FeedForwardNN,
        num_heads: int,
        prediction_strategy: str = ...,
    ) -> T: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> Tuple[Tensor, Tensor]: ...
    def set_head(self, new_head: int) -> None: ...
    def get_head(self) -> int: ...

class FelixNet(FeedForwardNN):
    _scale: nn.Linear
    def __init__(
        self, in_dim: Tuple, out_dim: Tuple, initial_scale: float = ...
    ) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> Tuple[Tensor, Tensor]: ...
