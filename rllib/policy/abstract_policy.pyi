from abc import ABC, abstractmethod
from torch import Tensor
from typing import Iterator
from rllib.dataset.datatypes import Observation, Distribution


class AbstractPolicy(ABC):
    dim_state: int
    dim_action: int
    num_states: int
    num_actions: int
    deterministic: bool

    def __init__(self, dim_state: int, dim_action: int,
                 num_states: int = None, num_actions: int = None,
                 deterministic: bool = False) -> None: ...

    @abstractmethod
    def __call__(self, state: Tensor) -> Distribution: ...

    @property  # type: ignore
    @abstractmethod
    def parameters(self) -> Iterator: ...

    @parameters.setter  # type: ignore
    @abstractmethod
    def parameters(self, new_params: Iterator) -> None: ...

    def random(self, batch_size: int = None) -> Distribution: ...

    def update(self, observation: Observation) -> None: ...

    @property
    def discrete_state(self) -> bool: ...

    @property
    def discrete_action(self) -> bool: ...
