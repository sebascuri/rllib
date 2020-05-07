from typing import Tuple, Callable, Union, Optional

from rllib.dataset.datatypes import State, Action, Reward, Done, Termination
from .abstract_environment import AbstractEnvironment
from .systems.abstract_system import AbstractSystem
from rllib.reward import AbstractReward


class SystemEnvironment(AbstractEnvironment):
    reward: AbstractReward
    system: AbstractSystem
    termination: Optional[Termination]
    initial_state: Callable[..., State]
    _time: float


    def __init__(self, system: AbstractSystem,
                 initial_state: Union[State, Callable[..., State]] = None,
                 reward: AbstractReward = None,
                 termination: Termination = None) -> None: ...

    def step(self, action: Action) -> Tuple[State, Reward, Done, dict]: ...

    def reset(self) -> State: ...

    def render(self, mode: str = 'human') -> None: ...

    @property
    def state(self) -> State: ...

    @state.setter
    def state(self, value: State) -> None: ...

    @property
    def time(self) -> float: ...
