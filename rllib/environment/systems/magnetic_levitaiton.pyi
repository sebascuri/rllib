from .ode_system import ODESystem
from .linear_system import LinearSystem
from rllib.dataset.datatypes import Action, State
import numpy as np


class PitchControl(ODESystem):
    mass: float
    resistance: float
    x_inf: float
    l_inf: float
    ksi: float
    max_action: float
    gravity: float

    def __init__(self, mass: float = 0.8, resistance: float = 11.68,
                 x_inf: float = 0.007, l_inf: float = 0.80502,
                 ksi: float = 0.001599, max_action: float = 60, gravity: float = 9.81,
                 step_size: float = 0.01) -> None: ...

    def linearize(self) -> LinearSystem: ...

    def alpha(self, x1: float, x2: float, x3: float) -> float: ...

    def beta(self, x1: float, x2: float, x3: float) -> float: ...

    def gamma(self, x1: float, x2: float, x3: float) -> float: ...

    def _ode(self, _: float, state: State, action: Action) -> State: ...