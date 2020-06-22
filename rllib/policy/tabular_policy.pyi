from typing import Type, TypeVar, Union

from torch import Tensor

from .nn_policy import NNPolicy

T = TypeVar("T", bound="NNPolicy")

class TabularPolicy(NNPolicy):
    def __init__(self, num_states: int, num_actions: int) -> None: ...
    @classmethod
    def from_other(cls: Type[T], other: T, copy: bool = ...) -> T: ...
    @property
    def table(self) -> Tensor: ...
    def set_value(self, state: Tensor, new_value: Union[Tensor, float]) -> None: ...
