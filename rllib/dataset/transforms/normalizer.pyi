from .abstract_transform import AbstractTransform
from .. import Observation
from numpy import ndarray
from torch import Tensor
from typing import Union

Array = Union[ndarray, Tensor]


class Normalizer(object):
    _mean: Array
    _variance: Array
    _count: int
    _preserve_origin: bool

    def __init__(self, preserve_origin: bool = False) -> None: ...

    def __call__(self, array: Array) -> Array: ...

    def inverse(self, array: Array) -> Array: ...

    def update(self, array: Array) -> None: ...


class StateNormalizer(AbstractTransform):
    _normalizer: Normalizer

    def __init__(self, preserve_origin: int = False) -> None: ...

    def __call__(self, observation: Observation) -> Observation: ...

    def inverse(self, observation: Observation) -> Observation: ...

    def update(self, observation: Observation) -> None: ...


class ActionNormalizer(AbstractTransform):
    _normalizer: Normalizer

    def __init__(self, preserve_origin: int = False) -> None: ...

    def __call__(self, observation: Observation) -> Observation: ...

    def inverse(self, observation: Observation) -> Observation: ...

    def update(self, observation: Observation) -> None: ...