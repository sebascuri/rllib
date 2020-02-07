from abc import ABC, abstractmethod
from torch.distributions import Categorical, MultivariateNormal
from typing import Union
import numpy as np

Distribution = Union[Categorical, MultivariateNormal]
Action = Union[np.ndarray, int]

def argmax(action_distribution: Distribution) -> np.ndarray: ...


class ParameterDecay(ABC):
    start: float
    end: float
    decay: float

    def __init__(self, start: float, end: float = None, decay: float = None
                 ) -> None: ...

    @abstractmethod
    def __call__(self, steps: int = None) -> float: ...


class ExponentialDecay(ParameterDecay):
    def __call__(self, steps: int = None) -> float: ...


class LinearDecay(ParameterDecay):
    def __call__(self, steps: int = None) -> float: ...
