from typing import Tuple, Callable, Union

from rllib.dataset.datatypes import State, Action, Reward, Done
from .abstract_environment import AbstractEnvironment
from .systems.abstract_system import AbstractSystem


class SystemEnvironment(AbstractEnvironment):
    reward: Callable[..., Reward]
    system: AbstractSystem
    termination: Callable[..., Done]
    initial_state: Callable[..., State]
    _time: float


    def __init__(self, system: AbstractSystem,
                 initial_state: Union[State, Callable[..., State]] = None,
                 reward: Callable[..., Reward] = None,
                 termination: Callable[..., Done] = None) -> None: ...

    def step(self, action: Action) -> Tuple[State, Reward, Done, dict]: ...

    def reset(self) -> State: ...

    def render(self, mode: str = 'human') -> None: ...

    @property
    def state(self) -> State: ...

    @state.setter
    def state(self, value: State) -> None: ...

    @property
    def time(self) -> float: ...
