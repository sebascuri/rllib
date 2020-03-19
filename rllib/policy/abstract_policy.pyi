from abc import ABCMeta
from typing import Iterator

import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter

from rllib.dataset.datatypes import Observation, TupleDistribution


class AbstractPolicy(nn.Module, metaclass=ABCMeta):
    dim_state: int
    dim_action: int
    num_states: int
    num_actions: int
    deterministic: bool
    tau: float
    discrete_state: bool
    discrete_action: bool

    def __init__(self, dim_state: int, dim_action: int,
                 num_states: int = None, num_actions: int = None, tau: float = 1.0,
                 deterministic: bool = False) -> None: ...

    def forward(self, *args: Tensor, **kwargs) -> TupleDistribution: ...

    def random(self, batch_size: int = None) -> TupleDistribution: ...

    def update(self, observation: Observation) -> None: ...

    def update_parameters(self, new_parameters: Iterator[Parameter]) -> None: ...

