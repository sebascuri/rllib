from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from gym.spaces import Space

from rllib.dataset.datatypes import Action, Done, Reward, State
from rllib.model.abstract_model import AbstractModel

class AbstractEnvironment(object, metaclass=ABCMeta):
    dim_action: Tuple[int]
    dim_state: Tuple[int]
    dim_observation: Tuple[int]
    dim_reward: Tuple[int]
    num_actions: int
    num_states: int
    num_observations: int
    discrete_state: bool
    discrete_action: bool
    discrete_observation: bool
    action_space: Space
    observation_space: Space
    metadata: Dict[str, List]
    def __init__(
        self,
        dim_state: Tuple[int],
        dim_action: Tuple[int],
        observation_space: Space,
        action_space: Space,
        dim_observation: Optional[Tuple] = ...,
        num_states: Optional[int] = ...,
        num_actions: Optional[int] = ...,
        num_observations: Optional[int] = ...,
        dim_reward: Tuple[int] = ...,
    ) -> None: ...
    def __str__(self) -> str: ...
    @abstractmethod
    def step(self, action: Action) -> Tuple[State, Reward, Done, dict]: ...
    @abstractmethod
    def reset(self) -> State: ...
    def render(self, mode: str = ...) -> Union[None, np.ndarray, str]: ...
    def close(self) -> None: ...
    @property
    def action_scale(self) -> Action: ...
    @property
    def goal(self) -> Union[None, State]: ...
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

class EnvironmentBuilder(object, metaclass=ABCMeta):
    @abstractmethod
    def create_environment(self) -> AbstractEnvironment: ...
    def get_dynamical_model(self) -> Optional[AbstractModel]: ...
    def get_reward_model(self) -> Optional[AbstractModel]: ...
    def get_termination_model(self) -> Optional[AbstractModel]: ...
