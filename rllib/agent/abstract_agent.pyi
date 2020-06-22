from abc import ABCMeta
from typing import Dict, List, Optional

from torch import Tensor

from rllib.dataset.datatypes import Action, Distribution, Observation, State
from rllib.policy import AbstractPolicy
from rllib.util.logger import Logger
from rllib.util.parameter_decay import ParameterDecay

class AbstractAgent(object, metaclass=ABCMeta):
    policy: AbstractPolicy
    pi: Distribution
    counters: Dict[str, int]
    episode_steps: List[int]
    logger: Logger
    gamma: float
    exploration_steps: int
    exploration_episodes: int
    num_rollouts: int
    train_frequency: int
    _training: bool
    comment: str
    dist_params: dict
    params: Dict[str, ParameterDecay]
    last_trajectory: List[Observation]
    def __init__(
        self,
        train_frequency: int,
        num_rollouts: int,
        gamma: float = ...,
        exploration_steps: int = ...,
        exploration_episodes: int = ...,
        tensorboard: bool = ...,
        comment: str = ...,
    ) -> None: ...
    def act(self, state: State) -> Action: ...
    def observe(self, observation: Observation) -> None: ...
    def start_episode(self) -> None: ...
    def end_episode(self) -> None: ...
    def end_interaction(self) -> None: ...
    def set_goal(self, goal: Optional[Tensor]) -> None: ...
    def _train(self) -> None: ...
    def train(self, val: bool = True) -> None: ...
    def eval(self, val: bool = True) -> None: ...
    @property
    def total_episodes(self) -> int: ...
    @property
    def total_steps(self) -> int: ...
    @property
    def train_steps(self) -> int: ...
    @property
    def name(self) -> str: ...
    def save(self, filename: str, directory: Optional[str] = ...) -> str: ...
    def load(self, path: str) -> None: ...
