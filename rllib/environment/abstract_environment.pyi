from abc import ABC, abstractmethod
from gym.spaces import Space
from typing import Tuple, Union
import numpy as np
from torch import Tensor

State = Union[np.ndarray, int, Tensor]
Action = Union[np.ndarray, int, Tensor]


class AbstractEnvironment(ABC):
    dim_action: int
    dim_state: int
    dim_observation: int
    num_actions: int
    num_states: int
    num_observations: int
    action_space: Space
    observation_space: Space

    @abstractmethod
    def step(self, action: Action) -> Tuple[State, float, bool, dict]: ...

    @abstractmethod
    def reset(self) -> State: ...

    def render(self, mode: str = 'human'): ...

    def close(self) -> None: ...

    @property  # type: ignore
    @abstractmethod
    def state(self) -> State: ...

    @state.setter  # type: ignore
    @abstractmethod
    def state(self, value: State) -> None: ...

    @property
    @abstractmethod
    def time(self) -> float: ...

    @property
    def discrete_state(self) -> bool: ...

    @property
    def discrete_action(self) -> bool: ...

    @property
    def discrete_observation(self) -> bool: ...

    @property
    def name(self) -> str: ...