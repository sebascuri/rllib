"""Implementation of different Neural Networks with pytorch."""

from typing import List, Dict, Tuple, TypeVar, Type, Any
import torch.nn as nn
from torch import Tensor

T = TypeVar('T', bound='FeedForwardNN')


class FeedForwardNN(nn.Module):
    kwargs: Dict
    embedding_dim: int
    hidden_layers: nn.Sequential
    head: nn.Linear
    squashed_output: bool

    def __init__(self, in_dim: int, out_dim: int, layers: List[int] = None,
                 non_linearity: str = 'ReLU', biased_head: bool = True,
                 squashed_output: bool = False) -> None: ...

    @classmethod
    def from_other(cls: Type[T], other: T, copy: bool = False) -> T: ...

    def forward(self, *args: Tensor, **kwargs) -> Any: ...

    def last_layer_embeddings(self, x: Tensor) -> Tensor: ...


class DeterministicNN(FeedForwardNN):
    def forward(self, *args: Tensor, **kwargs) -> Tensor: ...


class CategoricalNN(FeedForwardNN):
    def __init__(self, in_dim: int, out_dim: int, layers: List[int] = None,
                 non_linearity: str = 'ReLU', biased_head: bool = True) -> None: ...


class HeteroGaussianNN(FeedForwardNN):
    _covariance: nn.Linear

    def forward(self, *args: Tensor, **kwargs) -> Tuple[Tensor, Tensor]: ...


class HomoGaussianNN(FeedForwardNN):
    _covariance: nn.Parameter


    def forward(self, *args: Tensor, **kwargs) -> Tuple[Tensor, Tensor]: ...


class DeterministicEnsemble(FeedForwardNN):
    num_heads: int
    head_ptr: int

    def __init__(self, in_dim: int, out_dim: int, num_heads: int, layers: List[int] = None,
                 non_linearity: str = 'ReLU', biased_head: bool = True,
                 squashed_output: bool = False) -> None: ...

    @classmethod
    def from_feedforward(cls: Type[T], other: FeedForwardNN, num_heads: int) -> T: ...


    def forward(self, *args: Tensor, **kwargs) -> Tuple[Tensor, Tensor]: ...

    def select_head(self, new_head: int) -> None: ...


class FelixNet(nn.Module):
    layers: List[int]
    linear1: nn.Linear
    linear2: nn.Linear
    _mean: nn.Linear
    _covariance: nn.Linear

    def __init__(self, in_dim: int, out_dim: int) -> None: ...

    def forward(self, *args: Tensor, **kwargs) -> Tuple[Tensor, Tensor]: ...