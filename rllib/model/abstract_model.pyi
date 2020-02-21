from abc import ABCMeta, abstractmethod
from rllib.dataset.datatypes import State, Action, Distribution
from typing import Iterator, Union, List


class AbstractModel(object, metaclass=ABCMeta):
    dim_state: int
    dim_action: int
    dim_observation: int
    num_states: int
    num_actions: int
    num_observations: int

    def __init__(self, dim_state: int, dim_action: int, dim_observation: int = None,
                 num_states: int = None, num_actions: int = None,
                 num_observations: int = None) -> None: ...

    @abstractmethod
    def __call__(self, state: State, action: Action) -> Distribution: ...

    @property
    def parameters(self) -> Union[List[Iterator], Iterator]: ...

    @parameters.setter
    def parameters(self, new_params: Union[List[Iterator], Iterator]) -> None: ...

    @property
    def discrete_state(self) -> bool: ...

    @property
    def discrete_action(self) -> bool: ...

    @property
    def discrete_observation(self) -> bool: ...
