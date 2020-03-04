from .abstract_policy import AbstractPolicy
from rllib.dataset.datatypes import Distribution
from torch.distributions import Categorical
from typing import List, Union, Callable
from torch import Tensor
import torch.nn as nn

class NNPolicy(AbstractPolicy):
    input_transform: Callable[[Tensor], Tensor]
    nn: nn.Module

    def __init__(self, dim_state: int, dim_action: int,
                 num_states: int = None, num_actions: int = None,
                 layers: List[int] = None, biased_head: bool = True,
                 tau: float = 1., deterministic: bool = False,
                 squashed_output: bool = True,
                 input_transform: Callable[[Tensor], Tensor] = None) -> None: ...

    def forward(self, *args: Tensor, **kwargs) -> Distribution: ...

    def embeddings(self, state: Tensor, action: Tensor = None) -> Tensor: ...


class TabularPolicy(NNPolicy):
    def __init__(self, num_states: int, num_actions: int) -> None: ...

    def forward(self, *args: Tensor, **kwargs) -> Categorical: ...

    @property
    def table(self) -> Tensor: ...

    def set_value(self, state: Tensor, new_value: Union[Tensor, float]) -> None: ...


class FelixPolicy(AbstractPolicy):

    def __init__(self, dim_state: int, dim_action: int,
                 num_states: int = None, num_actions: int = None,
                 tau: float = 1., deterministic: bool = False) -> None: ...

    def forward(self, *args: Tensor, **kwargs) -> Distribution: ...
