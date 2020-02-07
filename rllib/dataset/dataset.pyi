from torch.utils import data
from numpy import ndarray
from . import Observation
from typing import List
from .transforms import AbstractTransform


class TrajectoryDataset(data.Dataset):
    _sequence_length: int
    _trajectories: List[Observation]
    _sub_trajectory_indexes: List [int]
    transformations: List[AbstractTransform]

    def __init__(self, sequence_length: int = None,
                 transformations: List[AbstractTransform] = None) -> None: ...

    def __getitem__(self, idx: int) -> Observation: ...

    def __len__(self) -> int: ...

    def append(self, trajectory: List[Observation]) -> None: ...

    def shuffle(self) -> None: ...

    @property
    def initial_states(self) -> ndarray: ...

    @property
    def sequence_length(self) -> int: ...

    @sequence_length.setter
    def sequence_length(self, value: int) -> None: ...

    @staticmethod
    def _get_subindexes(num_observations: int, sequence_length: int,
                        drop_last: bool = False) -> List[int]: ...