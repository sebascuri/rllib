from .abstract_transform import AbstractTransform
from .. import Observation


class RewardClipper(AbstractTransform):
    _min_reward: float
    _max_reward: float

    def __init__(self, min_reward: float = 0., max_reward: float = 1.) -> None: ...

    def __call__(self, observation: Observation) -> Observation: ...

    def inverse(self, observation: Observation) -> Observation: ...
