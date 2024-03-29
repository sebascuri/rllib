from typing import List

from rllib.util.utilities import MovingAverage

class EarlyStopping(object):
    epsilon: float
    relative: bool
    moving_average: List[MovingAverage]
    min_value: List[float]
    non_decrease_iter: int
    count: int
    total_count: int
    min_total_count: int
    def __init__(
        self,
        epsilon: float = ...,
        non_decrease_iter: int = ...,
        relative: bool = ...,
        min_total_count: int = ...,
    ) -> None: ...
    @property
    def stop(self) -> bool: ...
    def restart(self) -> None: ...
    def _reset(self, num: int, hard: bool) -> None: ...
    def reset(self, hard: bool = ...) -> None: ...
    def update(self, *args: float) -> None: ...
