"""An Offline dataset is intended for an offline rl algorithm to use."""

from typing import Dict, Iterator, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset

from .datatypes import Array, Index, Observation
from .transforms import AbstractTransform

T = TypeVar("T", bound="OfflineDataset")

class OfflineDataset(Dataset):
    """Offline dataset that handles the get item."""

    num_memory_steps: int
    indexes: torch.Tensor
    dataset: Observation
    transformations: List[AbstractTransform]
    bootstrap: bool
    max_len: int
    def __init__(
        self,
        dataset: Observation,
        transformations: Optional[Union[List[AbstractTransform], nn.ModuleList]] = ...,
        num_bootstraps: int = ...,
        bootstrap: float = ...,
        init_transformations: float = ...,
    ) -> None: ...
    @property
    def num_bootstraps(self) -> int: ...
    @num_bootstraps.setter
    def num_bootstraps(self, num_bootstraps: int) -> None: ...
    def __iter__(self) -> Iterator[Observation]: ...
    def __len__(self) -> int: ...
    def __getitem__(
        self, idx: Index
    ) -> Tuple[Dict[str, torch.Tensor], int, torch.Tensor]: ...
    def init_transformations(self) -> None: ...
    def apply_transformations(self, observation: Observation) -> Observation: ...
    def get_random_split(self, ratio: float) -> Tuple[Type[T], Type[T]]: ...
    def _get_raw_observation(self, idx: Index) -> Observation: ...
    def _get_observation(self, idx: Index) -> Observation: ...
    def sample_batch(
        self, batch_size: int
    ) -> Tuple[Observation, Index, torch.Tensor]: ...
    @property
    def all_data(self) -> Observation: ...
    @property
    def all_raw(self) -> Observation: ...
