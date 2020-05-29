from rllib.dataset.datatypes import Action, State

from .ode_system import ODESystem


class CartPole(ODESystem):
    pendulum_mass: float
    cart_mass: float
    length: float
    rot_friction: float
    gravity: float

    def __init__(self, pendulum_mass: float, cart_mass: float, length: float,
                 rot_friction: float = 0., gravity: float = 9.81,
                 step_size: float = 0.01) -> None: ...

    def _ode(self, _, state: State, action: Action) -> State: ...
