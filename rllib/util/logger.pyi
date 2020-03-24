"""Implementation of a Logger class."""

from typing import List, Dict, Tuple, Iterator, Union

import tensorboardX
import numpy as np
import torch

class Logger(object):
    statistics: List[Dict[str, float]]
    current: Dict[str, Tuple[int, float]]
    name: str
    writer: tensorboardX.SummaryWriter
    start: float

    def __init__(self, name: str) -> None: ...

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[Dict[str, float]]: ...

    def __getitem__(self, item: int) -> Dict[str, float]: ...

    def __str__(self) -> str: ...

    def get(self, key: str) -> List[float]: ...

    def keys(self) -> Iterator[str]: ...

    def update(self, **kwargs) -> None: ...

    def end_episode(self, **kwargs) -> None: ...

    def dump(self, name: str) -> None: ...
