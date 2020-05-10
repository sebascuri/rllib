from typing import List

import numpy as np

from .abstract_transform import AbstractTransform
from rllib.dataset.datatypes import Observation


class AngleWrapper(AbstractTransform):
    _indexes: List[int]

    def __init__(self, indexes: List[int]) -> None: ...

    def forward(self, *observation: Observation, **kwargs) -> Observation: ...

    def inverse(self, observation: Observation) -> Observation: ...
