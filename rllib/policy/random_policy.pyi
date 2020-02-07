from .abstract_policy import AbstractPolicy, Distribution
from typing import Iterator
from torch import Tensor


class RandomPolicy(AbstractPolicy):
    def __init__(self, dim_state: int, dim_action: int,
                 num_states: int = None, num_actions: int = None,
                 temperature: float = 1.) -> None: ...


    def __call__(self, state: Tensor) -> Distribution: ...

    @property
    def parameters(self) -> Iterator: ...


    @parameters.setter
    def parameters(self, new_params: Iterator) -> None: ...

