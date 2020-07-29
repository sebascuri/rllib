from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
from gym import spaces

from rllib.dataset.datatypes import Action, State

class AbstractSystem(object, metaclass=ABCMeta):
    dim_state: Tuple
    dim_action: Tuple
    dim_observation: Tuple
    _time: float
    def __init__(
        self,
        dim_state: Tuple,
        dim_action: Tuple,
        dim_observation: Optional[Tuple] = ...,
    ) -> None: ...
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
    def render(self, mode: str = ...) -> Union[None, np.ndarray, str]: ...
    @property
    def action_space(self) -> spaces.Space: ...
    @property
    def observation_space(self) -> spaces.Space: ...
