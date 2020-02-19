from .abstract_exploration_strategy import AbstractExplorationStrategy
from rllib.dataset.datatypes import State
import numpy as np

class GaussianNoise(AbstractExplorationStrategy):
    def __call__(self, state: State = None) -> np.ndarray: ...
