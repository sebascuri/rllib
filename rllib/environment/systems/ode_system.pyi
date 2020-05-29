from typing import Callable, Type

from scipy.integrate import RK45, OdeSolver

from rllib.dataset.datatypes import Action, State

from .abstract_system import AbstractSystem
from .linear_system import LinearSystem


class ODESystem(AbstractSystem):
    step_size: float
    func: Callable
    integrator: Type[OdeSolver]

    def __init__(self, func: Callable, step_size: float, dim_state: int, dim_action: int,
                 integrator: Type[OdeSolver] = RK45) -> None: ...

    def step(self, action: Action) -> State: ...

    def linearize(self, state: State=None, action: Action=None) -> LinearSystem: ...

    def reset(self, state: State = None) -> State: ...

    @property
    def state(self) -> State: ...

    @state.setter
    def state(self, value: State) -> None: ...

    @property
    def time(self) -> float: ...
