from abc import ABCMeta
from typing import List
import numpy as np
from rllib.dataset.datatypes import Observation, State, Action
from rllib.policy import AbstractPolicy


class AbstractAgent(object, metaclass=ABCMeta):
    policy: AbstractPolicy
    logs: dict
    gamma: float
    exploration_steps: int
    exploration_episodes: int

    def __init__(self, gamma: float = 1.0, exploration_steps: int = 0,
                 exploration_episodes: int = 0) -> None: ...

    def act(self, state: State) -> Action: ...

    def observe(self, observation: Observation) -> None: ...

    def start_episode(self) -> None: ...

    def end_episode(self) -> None: ...

    def end_interaction(self) -> None: ...

    @property
    def episodes_steps(self) -> List[int]: ...

    @property
    def episodes_rewards(self) -> List[List[float]]: ...

    @property
    def episodes_cumulative_rewards(self) -> List[float]: ...

    @property
    def total_episodes(self) -> int: ...

    @property
    def total_steps(self) -> int: ...

    @property
    def episode_steps(self) -> int: ...

    @property
    def name(self) -> str: ...
