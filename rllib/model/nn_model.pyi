"""Model implemented by a Neural Network."""
from typing import List, Optional

import torch
from torch import Tensor

from rllib.dataset.datatypes import TupleDistribution

from .abstract_model import AbstractModel

class NNModel(AbstractModel):
    input_transform: torch.nn.Module
    nn: torch.nn.Module
    deterministic: bool
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        num_states: Optional[int] = ...,
        num_actions: Optional[int] = ...,
        initial_scale: float = ...,
        layers: Optional[List[int]] = ...,
        biased_head: bool = ...,
        non_linearity: str = ...,
        input_transform: Optional[torch.nn.Module] = ...,
        deterministic: bool = ...,
    ) -> None: ...
    def forward(self, *args: Tensor, **kwargs) -> TupleDistribution: ...
