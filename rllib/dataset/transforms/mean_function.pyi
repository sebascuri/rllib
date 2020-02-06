from .abstract_transform import AbstractTransform
from .. import Observation
from typing import Callable


class MeanFunction(AbstractTransform):
    mean_function: Callable

    def __init__(self, mean_function: Callable) -> None: ...

    def __call__(self, observation: Observation) -> Observation: ...

    def inverse(self, observation: Observation) -> Observation: ...
