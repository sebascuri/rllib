"""Implementation of different Neural Networks with pytorch."""

import torch.nn as nn
from torch import Tensor
from typing import List, Union
from torch.distributions import MultivariateNormal, Categorical
from rllib.util.utilities import Delta

Gaussian = Union[MultivariateNormal, Delta]
Distribution = Union[MultivariateNormal, Delta, Categorical]


class DeterministicNN(nn.Module):
    layers = List[int]
    embedding_dim: int
    hidden_layers: nn.Sequential
    head: nn.Linear

    def __init__(self, in_dim: int, out_dim: int, layers: List[int] = None,
                 biased_head: bool = True) -> None: ...

    def forward(self, x: Tensor) -> Tensor: ...  # type: ignore

    def last_layer_embeddings(self, x: Tensor) -> Tensor: ...


class ProbabilisticNN(DeterministicNN):

    def __init__(self, in_dim: int, out_dim: int, layers: List[int] = None,
                 biased_head: bool = True) -> None: ...


class HeteroGaussianNN(ProbabilisticNN):
    _covariance: nn.Linear

    def __init__(self, in_dim: int, out_dim: int, layers: List[int] = None,
                 biased_head: bool = True) -> None: ...


    def forward(self, x: Tensor) -> Gaussian: ...  # type: ignore


class HomoGaussianNN(ProbabilisticNN):
    _covariance: nn.Linear

    def __init__(self, in_dim: int, out_dim: int, layers: List[int] = None,
                 biased_head: bool = True) -> None: ...

    def forward(self, x: Tensor) -> Gaussian: ...  # type: ignore


class CategoricalNN(ProbabilisticNN):

    def __init__(self, in_dim: int, out_dim: int, layers: List[int] = None,
                 biased_head: bool = True) -> None: ...

    def forward(self, x: Tensor) -> Categorical: ... # type: ignore


class EnsembleNN(ProbabilisticNN):
    num_heads: int

    def __init__(self, in_dim: int, out_dim: int, layers: List[int] = None,
                 biased_head: bool = True) -> None: ...

    def forward(self, x: Tensor) -> Gaussian: ...  # type: ignore


class FelixNet(nn.Module):
    layers: List[int]
    linear1: nn.Linear
    linear2: nn.Linear
    _mean: nn.Linear
    _covariance: nn.Linear

    def __init__(self, in_dim: int, out_dim: int) -> None: ...

    def forward(self, x: Tensor) -> Gaussian: ...  # type: ignore