import torch.nn as nn

from rllib.dataset.datatypes import Observation, Array
from .abstract_transform import AbstractTransform


class Scaler(nn.Module):
    _scale: float
    def __init__(self, scale: float) -> None: ...

    def forward(self, *array: Array, **kwargs) -> Array: ...

    def inverse(self, array: Array) -> Array: ...


class RewardScaler(AbstractTransform):
    _scaler: Scaler

    def __init__(self, scale: float) -> None: ...

    def forward(self, *observation: Observation, **kwargs) -> Observation: ...

    def inverse(self, observation: Observation) -> Observation: ...


class ActionScaler(AbstractTransform):
    _scaler: Scaler

    def __init__(self, scale: float) -> None: ...

    def forward(self, *observation: Observation, **kwargs) -> Observation: ...

    def inverse(self, observation: Observation) -> Observation: ...