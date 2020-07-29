from rllib.dataset.datatypes import Action, State

from .ode_system import ODESystem

class CartPole(ODESystem):
    pendulum_mass: float
    cart_mass: float
    length: float
    rot_friction: float
    gravity: float
    def __init__(
        self,
        pendulum_mass: float,
        cart_mass: float,
        length: float,
        rot_friction: float = ...,
        gravity: float = ...,
        step_size: float = ...,
    ) -> None: ...
    def _ode(self, _: float, state: State, action: Action) -> State: ...
