from typing import Callable, Dict, List, Optional, Tuple, Union

from rllib.dataset.datatypes import Action

from .abstract_environment import AbstractEnvironment

Transition = Dict[Tuple[int, int], List[Dict[str, Union[float, int]]]]

class MDP(AbstractEnvironment):
    _state: int
    _time: float
    transitions: Transition
    terminal_states: List[int]
    initial_state: Callable[..., int]
    def __init__(
        self,
        transitions: Transition,
        num_states: int,
        num_actions: int,
        initial_state: Optional[Union[int, Callable[..., int]]] = ...,
        terminal_states: Optional[List[int]] = ...,
    ): ...
    @property
    def state(self) -> int: ...
    @state.setter
    def state(self, value: int) -> None: ...
    def reset(self) -> int: ...
    @property
    def time(self) -> float: ...
    def step(self, action: Action) -> Tuple[int, float, bool, dict]: ...
    @staticmethod
    def check_transitions(
        transitions: Transition, num_states: int, num_actions: int
    ) -> None: ...
