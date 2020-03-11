from .ode_system import ODESystem
import numpy as np
from rllib.dataset.datatypes import Action, State
from .linear_system import LinearSystem
from gym.envs.classic_control.rendering import Viewer


class InvertedPendulum(ODESystem):
    mass: float
    length: float
    friction: float
    gravity: float
    step_size: float
    viewer: Viewer
    last_action: Action

    def __init__(self, mass: float, length: float, friction: float,
                 gravity: float = 9.81, step_size: float = 0.01) -> None: ...

    @property
    def inertia(self) -> float: ...

    def render(self, mode: str = 'human') -> np.ndarray: ...

    def _ode(self, _, state: State, action: Action) -> State: ...
