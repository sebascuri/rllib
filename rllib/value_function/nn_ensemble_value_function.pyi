from typing import List

import torch.nn as nn
from torch import Tensor

from .abstract_value_function import AbstractValueFunction, AbstractQFunction
from .nn_value_function import NNValueFunction, NNQFunction


class NNEnsembleValueFunction(AbstractValueFunction):
    dimension: int
    ensemble: nn.ModuleList

    def __init__(self, value_function: NNValueFunction = None,
                 dim_state: int = 1, num_states: int = None, layers: List[int] = None,
                 tau: float = 1.0, biased_head: bool=True, num_heads: int = 1) -> None: ...

    def __len__(self) -> int: ...

    def __getitem__(self, item: int) -> NNValueFunction: ...

    def forward(self, *args:Tensor, **kwargs) -> Tensor: ...

    def embeddings(self, state: Tensor) -> Tensor: ...


class NNEnsembleQFunction(AbstractQFunction):
    ensemble: nn.ModuleList

    def __init__(self, q_function: NNQFunction = None,
                 dim_state: int = 1, dim_action: int = 1,
                 num_states: int = None, num_actions: int = None,
                 layers: List[int] = None,  tau: float = 1.0, biased_head: bool = True,
                 num_heads: int = 1) -> None: ...

    def __len__(self) -> int: ...

    def __getitem__(self, item: int) -> NNValueFunction: ...

    def forward(self, *args: Tensor, **kwargs) -> Tensor: ...
