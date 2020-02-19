from .abstract_environment import AbstractEnvironment
import numpy as np
from torch import Tensor
from typing import List, Union, Callable, Tuple
from rllib.dataset.datatypes import State, Action

Array = Union[np.ndarray, Tensor]

class MDP(AbstractEnvironment):
    _state: Union[State, Callable[..., State]]
    _time: float
    kernel: Array
    reward: Array
    terminal_states: List[State]
    initial_state: Callable[..., State]

    def __init__(self, transition_kernel: Array, reward: Array,
                 initial_state: Union[State, Callable[..., State]] = None,
                 terminal_states: List[State] = None): ...

    @property
    def state(self) -> State: ...

    @state.setter
    def state(self, value: State) -> None: ...

    def reset(self) -> State: ...

    @property
    def time(self) -> float: ...

    def step(self, action: Action) -> Tuple[State, float, bool, dict]: ...
