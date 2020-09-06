from typing import Any, List, Optional

import torch.nn as nn
from torch import Tensor

from rllib.dataset.datatypes import Observation

from .abstract_transform import AbstractTransform

class NextStateClamper(AbstractTransform):
    lower: Tensor
    higher: Tensor
    def __init__(
        self, lower: Tensor, higher: Tensor, constant_idx: Optional[List[int]] = ...
    ) -> None: ...
    def forward(self, observation: Observation, **kwargs: Any) -> Observation: ...
    def inverse(self, observation: Observation) -> Observation: ...
