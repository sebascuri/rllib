from typing import List, Tuple, Type, TypeVar

from numpy import ndarray
from torch import Tensor
from torch.utils import data

from rllib.dataset.datatypes import Observation
from rllib.dataset.transforms import AbstractTransform


T = TypeVar('T', bound='ExperienceReplay')


class ExperienceReplay(data.Dataset):
    max_len: int
    batch_size: int
    memory: ndarray
    weights: Tensor
    transformations: List[AbstractTransform]
    _ptr: int

    def __init__(self, max_len: int, transformations: List[AbstractTransform] = None
                 ) -> None: ...

    @classmethod
    def from_other(cls: Type[T], other: T) -> T: ...

    def __getitem__(self, item: int) -> Tuple[Observation, int, Tensor]: ...

    def get_observation(self, idx: int) -> Observation: ...

    def __len__(self) -> int: ...

    def reset(self) -> None: ...

    def append(self, observation: Observation) -> None: ...

    def get_batch(self, batch_size: int) -> Tuple[Observation, Tensor, Tensor]: ...

    @property
    def all_data(self) -> Observation: ...

    @property
    def is_full(self) -> bool: ...

    def update(self, indexes: Tensor, td_error: Tensor) -> None: ...
