from typing import Optional

import numpy as np

from rllib.dataset.datatypes import Action, State

from .abstract_system import AbstractSystem

class LinearSystem(AbstractSystem):
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    _state: State
    def __init__(
        self, a: np.ndarray, b: np.ndarray, c: Optional[np.ndarray] = ...
    ) -> None: ...
    def step(self, action: Action) -> State: ...
    def reset(self, state: State) -> State: ...
    @property
    def state(self) -> State: ...
    @state.setter
    def state(self, value: State) -> None: ...
