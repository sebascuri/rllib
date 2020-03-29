from typing import List, Union, Callable, Tuple, Dict

from rllib.dataset.datatypes import Action, Array
from .abstract_environment import AbstractEnvironment


class MDP(AbstractEnvironment):
    _state: int
    _time: float
    transitions: Dict[Tuple[int, int], List]
    terminal_states: List[int]
    initial_state: Callable[..., int]

    def __init__(self, transitions: Dict[Tuple[int, int], List],
                 num_states: int, num_actions: int,
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

    @staticmethod
    def check_transitions(transitions: Dict[Tuple[int, int], List], num_states: int, num_actions: int
                      ) -> None: ...