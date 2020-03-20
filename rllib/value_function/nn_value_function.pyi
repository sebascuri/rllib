from typing import List, Union

import torch.nn
from torch import Tensor

from rllib.util.neural_networks import DeterministicNN
from .abstract_value_function import AbstractValueFunction, AbstractQFunction


class NNValueFunction(AbstractValueFunction):
    input_transform: torch.nn.Module
    dimension: int
    nn: DeterministicNN

    def __init__(self, dim_state: int, num_states: int = -1, layers: List[int] = None,
                 tau: float = 1.0, biased_head: bool=True,
                 input_transform: torch.nn.Module = None) -> None: ...

    def forward(self, *args: Tensor, **kwargs) -> Tensor: ...

    def embeddings(self, state: Tensor) -> Tensor: ...

class NNQFunction(AbstractQFunction):
    input_transform: torch.nn.Module
    nn: DeterministicNN
    tau: float

    def __init__(self, dim_state: int, dim_action: int,
                 num_states: int = -1, num_actions: int = -1,
                 layers: List[int] = None,  tau: float = 1.0, biased_head: bool=True,
                 input_transform: torch.nn.Module = None
                 ) -> None: ...

    def forward(self, *args: Tensor, **kwargs) -> Tensor: ...

