from .ode_system import ODESystem
from .linear_system import LinearSystem
from rllib.dataset.datatypes import Action, State
import numpy as np



class UnderwaterVehicle(ODESystem):

    def __init__(self, step_size: float = 0.01) -> None: ...

    def linearize(self) -> LinearSystem: ...

    def drag(self, velocity: float) -> float: ...

    def mass(self, velocity: float) -> float: ...

    def thrust(self, velocity: float, thrust: float) -> float: ...

    def _ode(self, _: float, state: State, action: Action) -> State: ...
