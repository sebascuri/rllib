from .ode_system import ODESystem
from .linear_system import LinearSystem
from .abstract_system import Action, State


class CartPole(ODESystem):
    pendulum_mass: float
    cart_mass: float
    length: float
    rot_friction: float
    gravity: float

    def __init__(self, pendulum_mass: float, cart_mass: float, length: float,
                 rot_friction: float = 0., gravity: float = 9.81,
                 step_size: float = 0.01) -> None: ...

    def linearize(self) -> LinearSystem: ...

    def _ode(self, _, state: State, action: Action) -> State: ...
