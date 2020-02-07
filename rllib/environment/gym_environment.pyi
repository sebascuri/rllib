from .abstract_environment import AbstractEnvironment, State, Action
import gym
from typing import Tuple


class GymEnvironment(AbstractEnvironment):
    env: gym.envs.registration
    _time: float

    def __init__(self, env_name: str, seed: int = None) -> None: ...

    def step(self, action: Action) -> Tuple[State, float, bool, dict]: ...

    def render(self, mode: str = 'human') -> None: ...

    def close(self) -> None: ...

    def reset(self) -> State: ...

    @property
    def state(self) -> State: ...

    @state.setter
    def state(self, value: State) -> None: ...

    @property
    def time(self) -> float: ...

    @property
    def name(self) -> str: ...