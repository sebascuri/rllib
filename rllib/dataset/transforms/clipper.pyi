from .abstract_transform import AbstractTransform
from rllib.dataset.datatypes import Observation, Array


class Clipper(object):
    _min: float
    _max: float

    def __init__(self, min_val: float, max_val: float) -> None: ...

    def __call__(self, array: Array) -> Array: ...

    def inverse(self, array: Array) -> Array: ...


class RewardClipper(AbstractTransform):
    _clipper: Clipper

    def __init__(self, min_reward: float = 0., max_reward: float = 1.) -> None: ...

    def __call__(self, observation: Observation) -> Observation: ...

    def inverse(self, observation: Observation) -> Observation: ...


class ActionClipper(AbstractTransform):
    _clipper: Clipper

    def __init__(self, min_action: float = 0., max_action: float = 1.) -> None: ...

    def __call__(self, observation: Observation) -> Observation: ...

    def inverse(self, observation: Observation) -> Observation: ...
