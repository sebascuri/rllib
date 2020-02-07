import numpy as np
from .abstract_system import AbstractSystem, State, Action


class LinearSystem(AbstractSystem):
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    _state: State

    def step(self, action: Action) -> State: ...

    def reset(self, state: State) -> State: ...

    @property
    def state(self) -> State: ...

    @state.setter
    def state(self, value: State) -> None: ...