from abc import ABCMeta
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import torch.nn as nn
from torch import Tensor

from rllib.dataset.datatypes import TupleDistribution
from rllib.environment.abstract_environment import AbstractEnvironment

T = TypeVar("T", bound="AbstractModel")

class AbstractModel(nn.Module, metaclass=ABCMeta):
    dim_state: Tuple[int]
    dim_action: Tuple[int]
    dim_observation: Tuple[int]
    dim_reward: Tuple[int]
    num_states: int
    num_actions: int
    num_observations: int
    discrete_state: bool
    discrete_action: bool
    model_kind: str
    goal: Optional[Tensor]
    temperature: Tensor
    allowed_model_kind: List[str]
    deterministic: bool
    _info: Dict[str, Any]
    def __init__(
        self,
        dim_state: Tuple[int],
        dim_action: Tuple[int],
        dim_observation: Tuple[int] = ...,
        dim_reward: Tuple[int] = ...,
        num_states: int = ...,
        num_actions: int = ...,
        num_observations: int = ...,
        model_kind: str = ...,
        goal: Optional[Tensor] = ...,
        deterministic: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    @classmethod
    def default(
        cls: Type[T], environment: AbstractEnvironment, *args: Any, **kwargs: Any
    ) -> T: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> TupleDistribution: ...
    @property
    def name(self) -> str: ...
    @property
    def info(self) -> Dict[str, Any]: ...
    def reset(self) -> None: ...
    def sample_posterior(self) -> None: ...
    def set_prediction_strategy(self, val: str) -> None: ...
    def scale(self, state: Tensor, action: Tensor) -> Tensor: ...
    def set_head(self, head_ptr: int) -> None: ...
    def get_head(self) -> int: ...
    def get_head_idx(self) -> Tensor: ...
    def get_prediction_strategy(self) -> str: ...
    def set_goal(self, goal: Optional[Tensor]) -> None: ...
    @property
    def is_rnn(self) -> bool: ...
