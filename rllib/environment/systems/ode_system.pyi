from typing import Callable, Optional, Tuple, Type

from scipy.integrate import OdeSolver

from rllib.dataset.datatypes import Action, State

from .abstract_system import AbstractSystem
from .linear_system import LinearSystem

class ODESystem(AbstractSystem):
    step_size: float
    func: Callable
    integrator: Type[OdeSolver]
    def __init__(
        self,
        func: Callable,
        step_size: float,
        dim_state: Tuple,
        dim_action: Tuple,
        integrator: Type[OdeSolver] = ...,
    ) -> None: ...
    def step(self, action: Action) -> State: ...
    def linearize(
        self, state: Optional[State] = ..., action: Optional[Action] = ...
    ) -> LinearSystem: ...
    def reset(self, state: Optional[State] = ...) -> State: ...
    @property
    def state(self) -> State: ...
    @state.setter
    def state(self, value: State) -> None: ...
    @property
    def time(self) -> float: ...
