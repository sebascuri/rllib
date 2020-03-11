from .abstract_transform import AbstractTransform
from rllib.dataset.datatypes import Observation, State, Action
from typing import Callable


class MeanFunction(AbstractTransform):
    mean_function: Callable[[State, Action], State]

    def __init__(self, mean_function: Callable[[State, Action], State]) -> None: ...

    def __call__(self, observation: Observation) -> Observation: ...

    def inverse(self, observation: Observation) -> Observation: ...
