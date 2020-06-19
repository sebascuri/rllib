"""Implementation of different Neural Networks with pytorch."""

from typing import Any, Dict, List, Tuple, Type, TypeVar, Optional

import torch.nn as nn
from torch import Tensor

T = TypeVar("T", bound="FeedForwardNN")

class FeedForwardNN(nn.Module):
    kwargs: Dict
    embedding_dim: int
    hidden_layers: nn.Sequential
    head: nn.Module
    squashed_output: bool
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        layers: Optional[List[int]] = ...,
        non_linearity: str = ...,
        biased_head: bool = ...,
        squashed_output: bool = ...,
    ) -> None: ...
    @classmethod
    def from_other(cls: Type[T], other: T, copy: bool = ...) -> T: ...
    def forward(self, *args: Tensor, **kwargs) -> Any: ...
    def last_layer_embeddings(self, x: Tensor) -> Tensor: ...

class DeterministicNN(FeedForwardNN):
    def forward(self, *args: Tensor, **kwargs) -> Tensor: ...

class CategoricalNN(FeedForwardNN):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        layers: Optional[List[int]] = ...,
        non_linearity: str = ...,
        biased_head: bool = ...,
    ) -> None: ...

class HeteroGaussianNN(FeedForwardNN):
    _scale: nn.Linear
    def forward(self, *args: Tensor, **kwargs) -> Tuple[Tensor, Tensor]: ...

class HomoGaussianNN(FeedForwardNN):
    _scale: nn.Parameter
    def forward(self, *args: Tensor, **kwargs) -> Tuple[Tensor, Tensor]: ...

class Ensemble(HeteroGaussianNN):
    num_heads: int
    head_ptr: int
    deterministic: bool
    prediction_strategy: str
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        prediction_strategy: str = ...,
        layers: Optional[List[int]] = ...,
        non_linearity: str = ...,
        biased_head: bool = ...,
        squashed_output: bool = ...,
        deterministic: bool = ...,
    ) -> None: ...
    @classmethod
    def from_feedforward(
        cls: Type[T],
        other: FeedForwardNN,
        num_heads: int,
        prediction_strategy: str = ...,
    ) -> T: ...
    def forward(self, *args: Tensor, **kwargs) -> Tuple[Tensor, Tensor]: ...
    def set_head(self, new_head: int) -> None: ...
    def get_head(self) -> int: ...

class FelixNet(FeedForwardNN):
    _scale: nn.Linear
    def __init__(self, in_dim: int, out_dim: int) -> None: ...
    def forward(self, *args: Tensor, **kwargs) -> Tuple[Tensor, Tensor]: ...
