from .abstract_system import AbstractSystem, State, Action
from typing import Callable
from scipy.integrate import OdeSolver


class ODESystem(AbstractSystem):
    step_size: float
    ode: OdeSolver

    def __init__(self, ode: Callable, step_size: float, dim_state: int, dim_action: int,
                 integrator: str = 'dopri5', jac: Callable=None) -> None: ...

    def step(self, action: Action) -> State: ...

    def reset(self, state: State = None) -> State: ...

    @property
    def state(self) -> State: ...

    @state.setter
    def state(self, value: State) -> None: ...

    @property
    def time(self) -> float: ...