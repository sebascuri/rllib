from typing import Optional, Type

from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from rllib.algorithms.reinforce import REINFORCE
from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractQFunction, AbstractValueFunction

from .on_policy_agent import OnPolicyAgent

class REINFORCEAgent(OnPolicyAgent):
    algorithm: REINFORCE
    policy_optimizer: Optimizer
    baseline_optimizer: Optimizer
    target_update_frequency: int
    num_iter: int
    def __init__(
        self,
        policy: AbstractPolicy,
        optimizer: Optimizer,
        baseline: AbstractValueFunction = None,
        critic: AbstractQFunction = None,
        criterion: Optional[Type[_Loss]] = None,
        num_iter: int = 1,
        target_update_frequency: int = 1,
        train_frequency: int = 0,
        num_rollouts: int = 1,
        gamma: float = 1.0,
        exploration_steps: int = 0,
        exploration_episodes: int = 0,
        tensorboard: bool = False,
        comment: str = "",
    ) -> None: ...
