from rllib.dataset.datatypes import Action, State

from .ode_system import ODESystem


class PitchControl(ODESystem):
    omega: float
    cld: float
    cmld: float
    cw: float
    cm: float
    eta: float

    def __init__(self, omega: float = 56.7, cld: float = 0.313, cmld: float = 0.0139,
                 cw: float = 0.232, cm: float = 0.426, eta: float = .0875,
                 step_size: float = 0.01) -> None: ...

    def _ode(self, _: float, state: State, action: Action) -> State: ...
