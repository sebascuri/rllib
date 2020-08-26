from abc import ABCMeta
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from torch import Tensor
from torch.optim.optimizer import Optimizer

from rllib.algorithms.abstract_algorithm import AbstractAlgorithm
from rllib.dataset.datatypes import Action, Distribution, Loss, Observation, State
from rllib.environment import AbstractEnvironment
from rllib.policy import AbstractPolicy
from rllib.util.logger import Logger
from rllib.util.parameter_decay import ParameterDecay

T = TypeVar("T", bound="AbstractAgent")

class AbstractAgent(object, metaclass=ABCMeta):
    policy: AbstractPolicy
    algorithm: AbstractAlgorithm
    optimizer: Optimizer
    pi: Distribution
    counters: Dict[str, int]
    episode_steps: List[int]
    logger: Logger
    gamma: float
    exploration_steps: int
    exploration_episodes: int
    num_rollouts: int
    num_iter: int
    batch_size: int
    train_frequency: int
    policy_update_frequency: int
    target_update_frequency: int
    clip_gradient_val: float

    _training: bool
    comment: str
    dist_params: dict
    params: Dict[str, ParameterDecay]
    last_trajectory: List[Observation]
    def __init__(
        self,
        optimizer: Optional[Optimizer] = ...,
        train_frequency: int = ...,
        num_rollouts: int = ...,
        num_iter: int = ...,
        batch_size: int = ...,
        policy_update_frequency: int = ...,
        target_update_frequency: int = ...,
        gamma: float = ...,
        exploration_steps: int = ...,
        exploration_episodes: int = ...,
        tensorboard: bool = ...,
        comment: str = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    @classmethod
    def default(
        cls: Type[T], environment: AbstractEnvironment, *args: Any, **kwargs: Any
    ) -> T: ...
    def act(self, state: State) -> Action: ...
    def observe(self, observation: Observation) -> None: ...
    def start_episode(self) -> None: ...
    def end_episode(self) -> None: ...
    def end_interaction(self) -> None: ...
    def set_goal(self, goal: Optional[Tensor]) -> None: ...
    def learn(self) -> None: ...
    def early_stop(self, *args: Any, **kwargs: Any) -> bool: ...
    def train(self, val: bool = True) -> None: ...
    def eval(self, val: bool = True) -> None: ...
    def _learn_steps(self, closure: Callable) -> Loss: ...
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
