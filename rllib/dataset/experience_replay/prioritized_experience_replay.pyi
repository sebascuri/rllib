from typing import List, Tuple

from numpy import ndarray

from rllib.dataset.datatypes import Observation
from rllib.dataset.transforms import AbstractTransform
from .experience_replay import ExperienceReplay


class PrioritizedExperienceReplay(ExperienceReplay):
    alpha: float
    beta: float
    epsilon: float
    beta_increment: float
    max_priority: float
    priorities: ndarray

    def __init__(self, max_len: int, alpha: float = 0.6, beta: float = 0.4,
                 epsilon: float = 0.01, beta_inc: float = 0.001,
                 batch_size: int = 1, max_priority: float = 10.,
                 transformations: List[AbstractTransform] = None
                 ) -> None: ...

    def append(self, observation: Observation) -> None: ...

    def update(self, indexes: ndarray, td_error: ndarray) -> None: ...

    def _get_priority(self, td_error: ndarray) -> ndarray: ...

    def get_batch(self, batch_size: int = None
                  ) -> Tuple[Observation, ndarray, ndarray]: ...

