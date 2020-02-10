from abc import ABC, abstractmethod
from torch import Tensor
from typing import Iterator


class AbstractValueFunction(ABC):
    dim_state: int
    num_states: int

    def __init__(self, dim_state: int, num_states: int = None) -> None: ...

    def __call__(self, state: Tensor, action: Tensor = None) -> Tensor: ...

    @property  # type: ignore
    @abstractmethod
    def parameters(self) -> Iterator: ...

    @parameters.setter  # type: ignore
    @abstractmethod
    def parameters(self, new_params: Iterator) -> None: ...

    @property
    def discrete_state(self) -> bool: ...


class AbstractQFunction(AbstractValueFunction):
    dim_action: int
    num_actions: int

    def __init__(self, dim_state: int, dim_action: int,
                 num_states: int = None, num_actions: int = None) -> None: ...

    @abstractmethod
    def __call__(self, state: Tensor, action: Tensor = None) -> Tensor: ...

    @property
    def discrete_action(self) -> bool: ...
