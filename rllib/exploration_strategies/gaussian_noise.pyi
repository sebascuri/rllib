from .abstract_exploration_strategy import AbstractExplorationStrategy
import numpy as np

class GaussianNoise(AbstractExplorationStrategy):
    def __call__(self) -> np.ndarray: ...
