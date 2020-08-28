from abc import ABCMeta
from typing import Any, Tuple, Type, TypeVar

import torch.nn as nn
from torch import Tensor

from rllib.environment import AbstractEnvironment

T = TypeVar("T", bound="AbstractQFunction")

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
    @classmethod
    def default(
        cls: Type[T], environment: AbstractEnvironment, *args: Any, **kwargs: Any
    ) -> T: ...

class AbstractValueFunction(AbstractQFunction):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
