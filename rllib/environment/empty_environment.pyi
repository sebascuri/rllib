from typing import Tuple

import numpy as np

from rllib.dataset.datatypes import Action, Done, Reward, State

from .abstract_environment import AbstractEnvironment

class EmptyEnvironment(AbstractEnvironment):
    """Dummy Environment for testing."""

    def __init__(
        self,
        dim_state: Tuple,
        dim_action: Tuple,
        num_states: int = ...,
        num_actions: int = ...,
    ) -> None: ...
    def step(self, action: Action) -> Tuple[State, Reward, Done, dict]: ...
    def reset(self) -> State: ...
    @property
    def state(self) -> State: ...
    @state.setter
    def state(self, value: State) -> None: ...
    @property
    def time(self) -> float: ...
