from abc import ABCMeta, abstractmethod
from typing import Tuple

from gym.spaces import Space

from rllib.dataset.datatypes import State, Action, Reward, Done


class AbstractEnvironment(object, metaclass=ABCMeta):
    dim_action: int
    dim_state: int
    dim_observation: int
    num_actions: int
    num_states: int
    num_observations: int
    action_space: Space
    observation_space: Space

    @abstractmethod
    def step(self, action: Action) -> Tuple[State, Reward, Done, dict]: ...

    @abstractmethod
    def reset(self) -> State: ...

    def render(self, mode: str = 'human') -> None: ...

    def close(self) -> None: ...

    @property  # type: ignore
    @abstractmethod
    def state(self) -> State: ...

    @state.setter  # type: ignore
    @abstractmethod
    def state(self, value: State) -> None: ...

    @property
    @abstractmethod
    def time(self) -> float: ...

    @property
    def discrete_state(self) -> bool: ...

    @property
    def discrete_action(self) -> bool: ...

    @property
    def discrete_observation(self) -> bool: ...

    @property
    def name(self) -> str: ...