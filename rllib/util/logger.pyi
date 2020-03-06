"""Implementation of a Logger class."""

import numpy as np
from typing import List, Union
Values = Union[float, int, np.ndarray]


class Logger(object):
    running_log: List[Values]
    episode_log: List[Values]
    current_log: List[Values]
    type_: str

    def __init__(self, type_: str) -> None: ...

    def start_episode(self) -> None: ...

    def append(self, value: Values) -> None: ...

    def end_episode(self) -> None: ...

    def dump(self, name: str) -> None: ...
