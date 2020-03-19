from abc import ABCMeta, abstractmethod
from rllib.dataset.datatypes import Observation
import torch.nn as nn


class AbstractTransform(nn.Module, metaclass=ABCMeta):

    def forward(self, *observation: Observation, **kwargs) -> Observation: ...

    def inverse(self, observation: Observation) -> Observation: ...

    def update(self, observation: Observation) -> None: ...
