from abc import ABCMeta
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
from typing import Iterator
from rllib.dataset.datatypes import Observation, Distribution


class AbstractPolicy(nn.Module, metaclass=ABCMeta):
    dim_state: int
    dim_action: int
    num_states: int
    num_actions: int
    deterministic: bool
    tau: float

    def __init__(self, dim_state: int, dim_action: int,
                 num_states: int = None, num_actions: int = None, tau: float = 1.0,
                 deterministic: bool = False) -> None: ...

    def forward(self, *args: Tensor, **kwargs) -> Distribution: ...

    def random(self, batch_size: int = None) -> Distribution: ...

    def update(self, observation: Observation) -> None: ...

    def update_parameters(self, new_parameters: Iterator[Parameter]) -> None: ...

    @property
    def discrete_state(self) -> bool: ...

    @property
    def discrete_action(self) -> bool: ...
