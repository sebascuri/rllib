import torch.nn as nn

from rllib.dataset.datatypes import Observation, Array
from .abstract_transform import AbstractTransform


class Clipper(nn.Module):
    _min: float
    _max: float

    def __init__(self, min_val: float, max_val: float) -> None: ...

    def forward(self, *array: Array, **kwargs) -> Array: ...

    def inverse(self, array: Array) -> Array: ...


class RewardClipper(AbstractTransform):
    _clipper: Clipper

    def __init__(self, min_reward: float = 0., max_reward: float = 1.) -> None: ...

    def forward(self, *observation: Observation, **kwargs) -> Observation: ...

    def inverse(self, observation: Observation) -> Observation: ...


class ActionClipper(AbstractTransform):
    _clipper: Clipper

    def __init__(self, min_action: float = 0., max_action: float = 1.) -> None: ...

    def forward(self, *observation: Observation, **kwargs) -> Observation: ...

    def inverse(self, observation: Observation) -> Observation: ...
