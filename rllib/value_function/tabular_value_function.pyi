from typing import Union

from torch import Tensor

from .nn_value_function import NNQFunction, NNValueFunction

class TabularValueFunction(NNValueFunction):
    def __init__(
        self, num_states: int, tau: float = 0.0, biased_head: bool = False
    ) -> None: ...
    @property
    def table(self) -> Tensor: ...
    def set_value(
        self, state: Union[Tensor, int], new_value: Union[Tensor, float]
    ) -> None: ...

class TabularQFunction(NNQFunction):
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        tau: float = 0.0,
        biased_head: bool = False,
    ) -> None: ...
    @property
    def table(self) -> Tensor: ...
    def set_value(
        self,
        state: Union[Tensor, int],
        action: Union[Tensor, int],
        new_value: Union[Tensor, float],
    ) -> None: ...
