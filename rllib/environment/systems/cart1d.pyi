"""Implementation of CarPole System."""

import numpy as np

from rllib.dataset.datatypes import Action, State
from rllib.environment.systems.ode_system import ODESystem

class Cart1d(ODESystem):
    max_action: float
    def __init__(self, step_size: float = ..., max_action: float = ...) -> None: ...
    def _ode(self, _: float, state: State, action: Action) -> State: ...
