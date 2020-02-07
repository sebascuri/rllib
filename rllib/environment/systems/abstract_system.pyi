from abc import ABC, abstractmethod
from gym import spaces
import numpy as np
from typing import Union

State = Union[np.ndarray, int]
Action = Union[np.ndarray, int]

class AbstractSystem(ABC):
    dim_state: int
    dim_action: int
    dim_observation: int
    _time: float
    def __init__(self, dim_state: int, dim_action: int, dim_observation: int = None
                 ) -> None :...

    @property  # type: ignore
    @abstractmethod
    def state(self) -> State: ...

    @state.setter  # type: ignore
    @abstractmethod
    def state(self, value: State) -> None: ...

    @property
    def time(self) -> float: ...

    @abstractmethod
    def step(self, action: Action) -> State: ...

    @abstractmethod
    def reset(self, state: State) -> State: ...

    @property
    def action_space(self) -> spaces.Space: ...

    @property
    def observation_space(self) -> spaces.Space: ...
