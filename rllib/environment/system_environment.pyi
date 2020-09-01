from typing import Callable, Optional, Tuple, Union

import numpy as np

from rllib.dataset.datatypes import Action, Done, Reward, State
from rllib.model import AbstractModel

from .abstract_environment import AbstractEnvironment
from .systems.abstract_system import AbstractSystem

class SystemEnvironment(AbstractEnvironment):
    reward: AbstractModel
    system: AbstractSystem
    termination_model: Optional[AbstractModel]
    initial_state: Callable[..., State]
    _time: float
    def __init__(
        self,
        system: AbstractSystem,
        initial_state: Optional[Union[State, Callable[..., State]]] = ...,
        reward: Optional[AbstractModel] = ...,
        termination_model: Optional[AbstractModel] = ...,
    ) -> None: ...
    def step(self, action: Action) -> Tuple[State, Reward, Done, dict]: ...
    def reset(self) -> State: ...
    def render(self, mode: str = ...) -> Union[None, np.ndarray, str]: ...
    @property
    def state(self) -> State: ...
    @state.setter
    def state(self, value: State) -> None: ...
    @property
    def time(self) -> float: ...
