from torch import Tensor
import torch.nn as nn

from rllib.dataset.datatypes import Observation, Array
from .abstract_transform import AbstractTransform


class Normalizer(nn.Module):
    mean: Tensor
    variance: Tensor
    count: Tensor
    preserve_origin: bool

    def __init__(self, preserve_origin: bool = False) -> None: ...

    def forward(self, *array, **kwargs) -> Tensor: ...

    def inverse(self, array: Tensor) -> Tensor: ...

    def update(self, array: Tensor) -> None: ...


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