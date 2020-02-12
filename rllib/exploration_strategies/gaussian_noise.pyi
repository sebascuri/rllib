from .abstract_exploration_strategy import AbstractExplorationStrategy, State
import numpy as np

class GaussianNoise(AbstractExplorationStrategy):
    def __call__(self, state: State=None) -> np.ndarray: ...
