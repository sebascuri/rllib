from typing import List, Type, Union

from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from rllib.algorithms.sarsa import SARSA
from rllib.dataset.datatypes import Observation
from rllib.policy import AbstractQFunctionPolicy
from rllib.value_function import AbstractQFunction

from .abstract_agent import AbstractAgent

class SARSAAgent(AbstractAgent):
    algorithm: SARSA
    policy: AbstractQFunctionPolicy
    optimizer: Optimizer
    target_update_frequency: int
    last_observation: Union[None, Observation]
    num_iter: int
    batch_size: int
    trajectory = List[Observation]
    def __init__(
        self,
        q_function: AbstractQFunction,
        policy: AbstractQFunctionPolicy,
        criterion: Type[_Loss],
        optimizer: Optimizer,
        num_iter: int = 1,
        batch_size: int = 1,
        target_update_frequency: int = 4,
        train_frequency: int = 0,
        num_rollouts: int = 1,
        gamma: float = 1.0,
        exploration_steps: int = 0,
        exploration_episodes: int = 0,
        tensorboard: bool = False,
        comment: str = "",
    ) -> None: ...
