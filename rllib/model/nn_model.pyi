"""Model implemented by a Neural Network."""
from typing import Any, List, Optional, Tuple

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
        initial_scale: float = ...,
        layers: Optional[List[int]] = ...,
        biased_head: bool = ...,
        non_linearity: str = ...,
        input_transform: Optional[torch.nn.Module] = ...,
        deterministic: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> TupleDistribution: ...
