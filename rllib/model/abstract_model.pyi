from abc import ABCMeta
from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn
from torch import Tensor

from rllib.dataset.datatypes import TupleDistribution

class AbstractModel(nn.Module, metaclass=ABCMeta):
    dim_state: Tuple
    dim_action: Tuple
    dim_observation: Tuple
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
        dim_state: Tuple,
        dim_action: Tuple,
        dim_observation: Tuple = ...,
        num_states: int = ...,
        num_actions: int = ...,
        num_observations: int = ...,
        model_kind: str = ...,
        goal: Optional[Tensor] = ...,
        deterministic: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> TupleDistribution: ...
    @property
    def name(self) -> str: ...
    @property
    def info(self) -> Dict[str, Any]: ...
    def sample_posterior(self) -> None: ...
    def set_prediction_strategy(self, val: str) -> None: ...
    def scale(self, state: Tensor, action: Tensor) -> Tensor: ...
    def set_head(self, head_ptr: int) -> None: ...
    def get_head(self) -> int: ...
    def get_head_idx(self) -> Tensor: ...
    def get_prediction_strategy(self) -> str: ...
    def set_goal(self, goal: Optional[Tensor]) -> None: ...
