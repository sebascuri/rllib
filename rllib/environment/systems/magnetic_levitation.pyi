from rllib.dataset.datatypes import Action, State

from .ode_system import ODESystem

class MagneticLevitation(ODESystem):
    mass: float
    resistance: float
    x_inf: float
    l_inf: float
    ksi: float
    max_action: float
    gravity: float
    def __init__(
        self,
        mass: float = ...,
        resistance: float = ...,
        x_inf: float = ...,
        l_inf: float = ...,
        ksi: float = ...,
        max_action: float = ...,
        gravity: float = ...,
        step_size: float = ...,
    ) -> None: ...
    def alpha(self, x1: float, x2: float, x3: float) -> float: ...
    def beta(self, x1: float, x2: float, x3: float) -> float: ...
    def gamma(self, x1: float, x2: float, x3: float) -> float: ...
    def _ode(self, _: float, state: State, action: Action) -> State: ...
