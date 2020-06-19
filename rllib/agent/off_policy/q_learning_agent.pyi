from typing import Type

from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from rllib.algorithms.q_learning import QLearning
from rllib.dataset import ExperienceReplay
from rllib.policy import AbstractQFunctionPolicy
from rllib.value_function import AbstractQFunction

from .off_policy_agent import OffPolicyAgent

class QLearningAgent(OffPolicyAgent):
    algorithm: QLearning
    policy: AbstractQFunctionPolicy
    def __init__(
        self,
        q_function: AbstractQFunction,
        policy: AbstractQFunctionPolicy,
        criterion: Type[_Loss],
        optimizer: Optimizer,
        memory: ExperienceReplay,
        num_iter: int = ...,
        batch_size: int = ...,
        target_update_frequency: int = ...,
        train_frequency: int = ...,
        num_rollouts: int = ...,
        gamma: float = ...,
        exploration_steps: int = ...,
        exploration_episodes: int = ...,
        tensorboard: bool = ...,
        comment: str = ...,
    ) -> None: ...
