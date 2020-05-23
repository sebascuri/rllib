from typing import List

from torch import Tensor
import torch.nn as nn

from rllib.dataset.datatypes import Observation, Array
from .abstract_transform import AbstractTransform


class NextStateClamper(AbstractTransform):
    lower: Tensor
    higher: Tensor

    def __init__(self, lower: Tensor, higher: Tensor, constant_idx: List[int] = None) -> None: ...

    def forward(self, *array, **kwargs) -> Observation: ...

    def inverse(self, array: Observation) -> Observation: ...
