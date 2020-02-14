from abc import ABC, abstractmethod
from rllib.policy import AbstractPolicy
from torch import Tensor
from typing import Iterator, Union, List


class AbstractValueFunction(ABC):
    dim_state: int
    num_states: int

    def __init__(self, dim_state: int, num_states: int = None) -> None: ...

    def __call__(self, state: Tensor, action: Tensor = None) -> Union[List[Tensor], Tensor]: ...

    @property  # type: ignore
    @abstractmethod
    def parameters(self) -> Union[List[Iterator], Iterator]: ...

    @parameters.setter  # type: ignore
    @abstractmethod
    def parameters(self, new_params: Union[List[Iterator], Iterator]) -> None: ...

    @property
    def discrete_state(self) -> bool: ...


class AbstractQFunction(AbstractValueFunction):
    dim_action: int
    num_actions: int

    def __init__(self, dim_state: int, dim_action: int,
                 num_states: int = None, num_actions: int = None) -> None: ...

    @abstractmethod
    def __call__(self, state: Tensor, action: Tensor = None) -> Union[List[Tensor], Tensor]: ...

    @property
    def discrete_action(self) -> bool: ...

    def value(self, state: Tensor, policy: AbstractPolicy, n_samples: int = 0) -> Union[List[Tensor], Tensor]: ...
