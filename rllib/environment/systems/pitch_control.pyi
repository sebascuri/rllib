from rllib.dataset.datatypes import Action, State

from .ode_system import ODESystem

class PitchControl(ODESystem):
    omega: float
    cld: float
    cmld: float
    cw: float
    cm: float
    eta: float
    def __init__(
        self,
        omega: float = ...,
        cld: float = ...,
        cmld: float = ...,
        cw: float = ...,
        cm: float = ...,
        eta: float = ...,
        step_size: float = ...,
    ) -> None: ...
    def _ode(self, _: float, state: State, action: Action) -> State: ...
