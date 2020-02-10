from abc import ABC
from typing import Union, List
import numpy as np
from torch import Tensor
from rllib.dataset import Observation
from rllib.policy import AbstractPolicy

State = Union[np.ndarray, int, Tensor]
Action = Union[np.ndarray, int, Tensor]
Reward = Union[np.ndarray, float, Tensor]
Done = Union[np.ndarray, bool, Tensor]

class AbstractAgent(ABC):
    num_episodes: int
    training: bool
    episode_length: int
    logs: dict
    gamma: float
    policy: AbstractPolicy

    def __init__(self, gamma: float = 1.0, episode_length: int = None) -> None: ...

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
    def total_steps(self) -> int: ...

    @property
    def episode_steps(self) -> int: ...

    @property
    def name(self) -> str: ...

    def train(self, mode: bool = True) -> None: ...

    def eval(self) -> None: ...
