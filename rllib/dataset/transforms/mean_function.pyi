from .abstract_transform import AbstractTransform
from rllib.dataset.datatypes import Observation
import torch.nn as nn


class MeanFunction(AbstractTransform):
    mean_function: nn.Module

    def __init__(self, mean_function: nn.Module) -> None: ...

    def forward(self, *observation: Observation, **kwargs) -> Observation: ...

    def inverse(self, observation: Observation) -> Observation: ...
