from abc import ABCMeta

import torch.nn as nn

from rllib.dataset.datatypes import Observation

class AbstractTransform(nn.Module, metaclass=ABCMeta):
    def forward(self, *observation: Observation, **kwargs) -> Observation: ...
    def inverse(self, observation: Observation) -> Observation: ...
    def update(self, observation: Observation) -> None: ...
