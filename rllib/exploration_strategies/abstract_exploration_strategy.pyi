from abc import ABCMeta, abstractmethod
from typing import Union

import numpy as np

from rllib.dataset.datatypes import State
from rllib.util import ParameterDecay


class AbstractExplorationStrategy(object, metaclass=ABCMeta):
    param: ParameterDecay
    dimension: int

    def __init__(self, param: Union[float, ParameterDecay], dimension: int = 1) -> None: ...

    @abstractmethod
    def __call__(self, state: State = None) -> np.ndarray: ...
