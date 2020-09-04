from abc import ABCMeta
from typing import Any, Dict, Optional, Tuple, Type, TypeVar

import torch.nn as nn
from torch import Tensor

from rllib.dataset.datatypes import Action, TupleDistribution
from rllib.environment import AbstractEnvironment

T = TypeVar("T", bound="AbstractPolicy")

class AbstractPolicy(nn.Module, metaclass=ABCMeta):
    dim_state: Tuple
    dim_action: Tuple
    num_states: int
    num_actions: int
    _deterministic: bool
    tau: float
    discrete_state: bool
    discrete_action: bool
    action_scale: Tensor
    goal: Optional[Tensor]
    dist_params: Dict[str, Any]
    def __init__(
        self,
        dim_state: Tuple,
        dim_action: Tuple,
        num_states: int = ...,
        num_actions: int = ...,
        tau: float = ...,
        deterministic: bool = ...,
        action_scale: Action = ...,
        goal: Optional[Tensor] = ...,
        dist_params: Optional[Dict[str, Any]] = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    @property
    def deterministic(self) -> bool: ...
    @deterministic.setter
    def deterministic(self, value: bool) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> TupleDistribution: ...
    def random(
        self, batch_size: Optional[Tuple[int]] = ..., normalized: bool = ...
    ) -> TupleDistribution: ...
    def reset(self) -> None: ...
    def update(self) -> None: ...
    def set_goal(self, goal: Optional[Tensor]) -> None: ...
    @classmethod
    def default(
        cls: Type[T], environment: AbstractEnvironment, *args: Any, **kwargs: Any
    ) -> T: ...
