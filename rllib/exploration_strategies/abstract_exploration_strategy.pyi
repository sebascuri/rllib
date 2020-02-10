from abc import ABC, abstractmethod
import numpy as np
from .utilities import Action, Distribution
from rllib.util import ParameterDecay


class AbstractExplorationStrategy(ABC):
    param: ParameterDecay

    def __init__(self, start: float, end: float = None, decay: float = None
                 ) -> None: ...

    @abstractmethod
    def __call__(self, action_distribution: Distribution, steps: int = None
                 ) -> Action: ...
