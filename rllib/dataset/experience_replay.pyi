from numpy import ndarray
from . import Observation
from .transforms import AbstractTransform
from typing import List, Tuple
from torch.utils import data


class ExperienceReplay(data.Dataset):
    max_len: int
    batch_size: int
    memory: ndarray
    _ptr: int
    _transformations: List[AbstractTransform]

    def __init__(self, max_len: int, batch_size: int = 1,
                 transformations: List[AbstractTransform] = None
                 ) -> None: ...

    def __getitem__(self, idx: int) -> Observation: ...

    def __len__(self) -> int: ...

    def append(self, observation: Observation) -> None: ...

    def get_batch(self, batch_size: int = None
                  ) -> Tuple[Observation, ndarray, ndarray]: ...

    @property
    def is_full(self) -> bool: ...

    @property
    def has_batch(self) -> bool: ...

    def update(self, indexes: ndarray, td_error: ndarray) -> None: ...


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

class LinfSampler(ExperienceReplay):
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


class L1Sampler(LinfSampler):

    def get_batch(self, batch_size: int = None
                  ) -> Tuple[Observation, ndarray, ndarray]: ...