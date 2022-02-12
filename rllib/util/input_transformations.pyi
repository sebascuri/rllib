"""Input transformations for spot."""
from abc import ABC, abstractmethod
from typing import Iterable

from torch import Tensor

class AbstractTransform(ABC):
    """Abstract Transformation definition."""

    extra_dim: int
    @abstractmethod
    def __call__(self, state: Tensor) -> Tensor: ...

class ComposeTransforms(AbstractTransform):
    transforms: Iterable[AbstractTransform]
    def __init__(self, transforms: Iterable[AbstractTransform]) -> None: ...
    def __call__(self, x: Tensor) -> Tensor: ...

class AngleToCosSin(AbstractTransform):
    transforms: Iterable[AbstractTransform]
    def __call__(self, x: Tensor) -> Tensor: ...

class RemoveAngle(AbstractTransform):
    transforms: Iterable[AbstractTransform]
    def __call__(self, x: Tensor) -> Tensor: ...

class RemovePosition(AbstractTransform):
    transforms: Iterable[AbstractTransform]
    def __call__(self, x: Tensor) -> Tensor: ...
