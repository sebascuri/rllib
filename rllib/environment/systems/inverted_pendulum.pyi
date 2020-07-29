from typing import Union

import numpy as np
from gym.envs.classic_control.rendering import Viewer

from rllib.dataset.datatypes import Action, State

from .ode_system import ODESystem

class InvertedPendulum(ODESystem):
    mass: float
    length: float
    friction: float
    gravity: float
    step_size: float
    viewer: Viewer
    last_action: Action
    def __init__(
        self,
        mass: float,
        length: float,
        friction: float,
        gravity: float = ...,
        step_size: float = ...,
    ) -> None: ...
    @property
    def inertia(self) -> float: ...
    def render(self, mode: str = ...) -> Union[None, np.ndarray]: ...
    def _ode(self, _: float, state: State, action: Action) -> State: ...
