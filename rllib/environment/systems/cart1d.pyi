"""Implementation of CarPole System."""

import numpy as np

from rllib.environment.systems.ode_system import ODESystem
from rllib.dataset.datatypes import Action, State


class Cart1d(ODESystem):
    max_action: float

    def __init__(self, step_size: float = 0.01, max_action: float = 1.) -> None: ...

    def _ode(self, _, state: State, action: Action) -> State: ...


