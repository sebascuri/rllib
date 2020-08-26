from abc import ABCMeta
from typing import Any, Tuple

import torch.nn as nn
from torch import Tensor

class AbstractQFunction(nn.Module, metaclass=ABCMeta):
    dim_action: Tuple
    num_actions: int
    discrete_action: bool
    dim_state: Tuple
    num_states: int
    tau: float
    discrete_state: bool
    def __init__(
        self,
        dim_state: Tuple,
        dim_action: Tuple,
        num_states: int = ...,
        num_actions: int = ...,
        tau: float = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> Tensor: ...

class AbstractValueFunction(AbstractQFunction):
    def __init__(
        self, dim_state: Tuple, num_states: int = ..., tau: float = ...
    ) -> None: ...
