from .abstract_policy import AbstractPolicy
from rllib.dataset.datatypes import TupleDistribution
from typing import List, Union
from torch import Tensor
import torch

class NNPolicy(AbstractPolicy):
    input_transform: torch.nn.Module
    nn: torch.nn.Module

    def __init__(self, dim_state: int, dim_action: int,
                 num_states: int = None, num_actions: int = None,
                 layers: List[int] = None, biased_head: bool = True,
                 tau: float = 1., deterministic: bool = False,
                 squashed_output: bool = True,
                 input_transform: torch.nn.Module = None) -> None: ...

    def forward(self, *args: Tensor, **kwargs) -> TupleDistribution: ...

    def embeddings(self, state: Tensor, action: Tensor = None) -> Tensor: ...


class TabularPolicy(NNPolicy):
    def __init__(self, num_states: int, num_actions: int) -> None: ...

    def forward(self, *args: Tensor, **kwargs) -> Tensor: ...

    @property
    def table(self) -> Tensor: ...

    def set_value(self, state: Tensor, new_value: Union[Tensor, float]) -> None: ...


class FelixPolicy(AbstractPolicy):

    def __init__(self, dim_state: int, dim_action: int,
                 num_states: int = None, num_actions: int = None,
                 tau: float = 1., deterministic: bool = False) -> None: ...

    def forward(self, *args: Tensor, **kwargs) -> TupleDistribution: ...
