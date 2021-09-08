from abc import ABCMeta
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from torch import Tensor
from torch.distributions import Distribution
from torch.optim.optimizer import Optimizer

from rllib.algorithms.abstract_algorithm import AbstractAlgorithm
from rllib.dataset.datatypes import Action, Loss, Observation, State
from rllib.environment import AbstractEnvironment
from rllib.policy import AbstractPolicy
from rllib.util.early_stopping import EarlyStopping
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
    early_stopping_algorithm: EarlyStopping
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
    device: str

    training: bool
    _training_verbose: bool
    comment: str
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
        early_stopping_epsilon: float = ...,
        gamma: float = ...,
        exploration_steps: int = ...,
        exploration_episodes: int = ...,
        tensorboard: bool = ...,
        comment: str = ...,
        training_verbose: bool = ...,
        device: str = ...,
        log_dir: Optional[str] = ...,
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
    def set_policy(self, new_policy: AbstractPolicy) -> None: ...
    def learn(self, *args: Any, **kwargs: Any) -> None: ...
    def early_stop(self, losses: Loss, **kwargs: Any) -> bool: ...
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
    def train_at_observe(self) -> bool: ...
    @property
    def train_at_end_episode(self) -> bool: ...
    @property
    def name(self) -> str: ...
    def save_checkpoint(self) -> None: ...
    def save(self, filename: str, directory: Optional[str] = ...) -> str: ...
    def load(self, path: str) -> None: ...
