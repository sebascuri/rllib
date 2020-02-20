from abc import ABCMeta, abstractmethod
import numpy as np
from rllib.util import ParameterDecay
from rllib.dataset.datatypes import State

class AbstractExplorationStrategy(object, metaclass=ABCMeta):
    param: ParameterDecay
    max_value: float
    dimension: int

    def __init__(self, start: float, end: float = None, decay: float = None,
                 max_value: float = None, dimension: int = 1) -> None: ...

    @abstractmethod
    def __call__(self, state: State = None) -> np.ndarray: ...
