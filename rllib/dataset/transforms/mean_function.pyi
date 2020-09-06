from typing import Any

import torch.nn as nn

from rllib.dataset.datatypes import Observation

from .abstract_transform import AbstractTransform

class DeltaState(nn.Module): ...

class MeanFunction(AbstractTransform):
    mean_function: nn.Module
    def __init__(self, mean_function: nn.Module) -> None: ...
    def forward(self, observation: Observation, **kwargs: Any) -> Observation: ...
    def inverse(self, observation: Observation) -> Observation: ...
