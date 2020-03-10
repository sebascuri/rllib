from abc import ABCMeta, abstractmethod
import numpy as np
from rllib.util import ParameterDecay
from rllib.dataset.datatypes import State
from typing import Union


class AbstractExplorationStrategy(object, metaclass=ABCMeta):
    param: ParameterDecay
    dimension: int

    def __init__(self, param: Union[float, ParameterDecay], dimension: int = 1) -> None: ...

    @abstractmethod
    def __call__(self, state: State = None) -> np.ndarray: ...
