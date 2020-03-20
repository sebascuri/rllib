import torch

from rllib.dataset.datatypes import State
from .abstract_exploration_strategy import AbstractExplorationStrategy


class GaussianNoise(AbstractExplorationStrategy):
    def __call__(self, state: State = None) -> torch.Tensor: ...
