from abc import ABCMeta
from typing import Tuple

import torch.nn as nn
from torch import Tensor

from rllib.dataset.datatypes import TupleDistribution


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
                 num_states: int = -1, num_actions: int = -1, tau: float = 0.0,
                 deterministic: bool = False) -> None: ...

    def forward(self, *args: Tensor, **kwargs) -> TupleDistribution: ...

    def random(self, batch_size: Tuple[int] = None) -> TupleDistribution: ...

    def reset(self) -> None: ...

    def update(self) -> None: ...

