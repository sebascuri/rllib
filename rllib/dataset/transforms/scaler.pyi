from .abstract_transform import AbstractTransform
from rllib.dataset.datatypes import Observation, Array
import numpy as np


class Scaler(object):
    _scale: float
    def __init__(self, scale: float) -> None: ...

    def __call__(self, array: Array) -> Array: ...

    def inverse(self, array: Array) -> Array: ...


class RewardScaler(AbstractTransform):
    _scaler: Scaler

    def __init__(self, scale: float) -> None: ...

    def __call__(self, observation: Observation) -> Observation: ...

    def inverse(self, observation: Observation) -> Observation: ...


class ActionScaler(AbstractTransform):
    _scaler: Scaler

    def __init__(self, scale: float) -> None: ...

    def __call__(self, observation: Observation) -> Observation: ...

    def inverse(self, observation: Observation) -> Observation: ...