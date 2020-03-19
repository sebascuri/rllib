from typing import List, Tuple

from numpy import ndarray
from torch.utils import data

from rllib.dataset.datatypes import Observation
from rllib.dataset.transforms import AbstractTransform


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
