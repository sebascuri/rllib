from typing import Any, List

import numpy as np

from rllib.dataset.datatypes import Observation

from .abstract_transform import AbstractTransform

class AngleWrapper(AbstractTransform):
    _indexes: List[int]
    def __init__(self, indexes: List[int]) -> None: ...
    def forward(self, observation: Observation, **kwargs: Any) -> Observation: ...
    def inverse(self, observation: Observation) -> Observation: ...
