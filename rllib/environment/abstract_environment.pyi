from abc import ABCMeta, abstractmethod
from typing import Tuple

from gym.spaces import Space

from rllib.dataset.datatypes import Action, Done, Reward, State

class AbstractEnvironment(object, metaclass=ABCMeta):
    dim_action: int
    dim_state: int
    dim_observation: int
    num_actions: int
    num_states: int
    num_observations: int
    discrete_state: bool
    discrete_action: bool
    discrete_observation: bool
    action_space: Space
    observation_space: Space
    def __str__(self) -> str: ...
    @abstractmethod
    def step(self, action: Action) -> Tuple[State, Reward, Done, dict]: ...
    @abstractmethod
    def reset(self) -> State: ...
    def render(self, mode: str = "human") -> None: ...
    def close(self) -> None: ...
    @property
    def action_scale(self) -> Action: ...
    @property
    def goal(self) -> State: ...
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
    def name(self) -> str: ...
