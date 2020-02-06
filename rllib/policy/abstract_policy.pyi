from abc import ABC, abstractmethod
from torch import Tensor
from torch.distributions import Distribution
from typing import Iterator


class AbstractPolicy(ABC):
    dim_state: int
    dim_action: int
    num_states: int
    num_actions: int
    temperature: float

    def __init__(self, dim_state: int, dim_action: int,
                 num_states: int = None, num_actions: int = None,
                 temperature: float = 1.) -> None: ...

    @abstractmethod
    def __call__(self, state: Tensor) -> Distribution: ...

    @property  # type: ignore
    @abstractmethod
    def parameters(self) -> Iterator: ...

    @parameters.setter  # type: ignore
    @abstractmethod
    def parameters(self, new_params: Iterator) -> None: ...

    def random(self, batch_size: int = None) -> Distribution: ...

    @property
    def discrete_state(self) -> bool: ...

    @property
    def discrete_action(self) -> bool: ...
