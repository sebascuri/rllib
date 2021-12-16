"""Model implemented by a Neural Network."""
from typing import Any, List, Optional, Sequence

import torch
from torch import Tensor

from rllib.dataset.datatypes import TupleDistribution
from rllib.util.input_transformations import AbstractTransform

from .abstract_model import AbstractModel

class NNModel(AbstractModel):
    input_transform: Optional[AbstractTransform]
    nn: torch.nn.ModuleList
    deterministic: bool
    def __init__(
        self,
        initial_scale: float = ...,
        layers: Sequence[int] = ...,
        biased_head: bool = ...,
        non_linearity: str = ...,
        input_transform: Optional[AbstractTransform] = ...,
        deterministic: bool = ...,
        per_coordinate: bool = ...,
        jit_compile: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def state_actions_to_input_data(self, state: Tensor, action: Tensor) -> Tensor: ...
    def stack_predictions(self, prediction_list: List[Tensor]) -> Tensor: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> TupleDistribution: ...
