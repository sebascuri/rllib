import torch.nn as nn

from rllib.dataset.datatypes import Observation, Array
from .abstract_transform import AbstractTransform


class Normalizer(nn.Module):
    _mean: Array
    _variance: Array
    _count: int
    _preserve_origin: bool

    def __init__(self, preserve_origin: bool = False) -> None: ...

    def forward(self, *array: Array, **kwargs) -> Array: ...

    def inverse(self, array: Array) -> Array: ...

    def update(self, array: Array) -> None: ...


class StateActionNormalizer(AbstractTransform):
    _state_normalizer: StateNormalizer
    _action_normalizer: ActionNormalizer

    def __init__(self, preserve_origin: bool = False) -> None: ...

    def forward(self, *observation: Observation, **kwargs) -> Observation: ...

    def inverse(self, observation: Observation) -> Observation: ...

    def update(self, observation: Observation) -> None: ...



class StateNormalizer(AbstractTransform):
    _normalizer: Normalizer

    def __init__(self, preserve_origin: bool = False) -> None: ...

    def forward(self, *observation: Observation, **kwargs) -> Observation: ...

    def inverse(self, observation: Observation) -> Observation: ...

    def update(self, observation: Observation) -> None: ...


class ActionNormalizer(AbstractTransform):
    _normalizer: Normalizer

    def __init__(self, preserve_origin: bool = False) -> None: ...

    def forward(self, *observation: Observation, **kwargs) -> Observation: ...

    def inverse(self, observation: Observation) -> Observation: ...

    def update(self, observation: Observation) -> None: ...