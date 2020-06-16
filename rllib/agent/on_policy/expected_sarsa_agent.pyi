from typing import Type

from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from rllib.algorithms.esarsa import ESARSA
from rllib.policy import AbstractQFunctionPolicy
from rllib.value_function import AbstractQFunction

from .on_policy_agent import OnPolicyAgent

class ExpectedSARSAAgent(OnPolicyAgent):
    algorithm: ESARSA
    policy: AbstractQFunctionPolicy
    def __init__(
        self,
        q_function: AbstractQFunction,
        policy: AbstractQFunctionPolicy,
        criterion: Type[_Loss],
        optimizer: Optimizer,
        num_inter: int = 1,
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
