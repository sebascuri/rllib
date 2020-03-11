from numpy import ndarray
from rllib.dataset.datatypes import Observation
from rllib.dataset.transforms import AbstractTransform
from typing import List, Tuple
from .experience_replay import ExperienceReplay


class EXP3Sampler(ExperienceReplay):
    eta: float
    beta: float
    max_priority: float
    priorities: ndarray

    def __init__(self, max_len: int, eta: float = 0.1, beta: float = 0.1,
                 batch_size: int = 1, max_priority: float = 1.,
                 transformations: List[AbstractTransform] = None
                 ) -> None: ...

    def append(self, observation: Observation) -> None: ...

    def update(self, indexes: ndarray, td_error: ndarray) -> None: ...

    def probabilities(self, indexes: ndarray = None, sign: int = 1): ...

    def get_batch(self, batch_size: int = None
                  ) -> Tuple[Observation, ndarray, ndarray]: ...
