from .abstract_environment import AbstractEnvironment
import numpy as np
from typing import List, Union, Callable, Tuple
from rllib.dataset.datatypes import Action, Array


class MDP(AbstractEnvironment):
    _state: int
    _time: float
    kernel: Array
    reward: Array
    terminal_states: List[int]
    initial_state: Callable[..., int]

    def __init__(self, transition_kernel: Array, reward: Array,
                 initial_state: Union[int, Callable[..., int]] = None,
                 terminal_states: List[int] = None): ...

    @property
    def state(self) -> int: ...

    @state.setter
    def state(self, value: int) -> None: ...

    def reset(self) -> int: ...

    @property
    def time(self) -> float: ...

    def step(self, action: Action) -> Tuple[int, float, bool, dict]: ...
