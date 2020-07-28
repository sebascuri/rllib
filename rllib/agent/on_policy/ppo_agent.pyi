from typing import Type

from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from rllib.algorithms.ppo import PPO
from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractValueFunction

from .on_policy_agent import OnPolicyAgent

class PPOAgent(OnPolicyAgent):

    algorithm: PPO
    target_kl: float
    def __init__(
        self,
        policy: AbstractPolicy,
        value_function: AbstractValueFunction,
        optimizer: Optimizer,
        criterion: Type[_Loss],
        epsilon: float = ...,
        lambda_: float = ...,
        target_kl: float = ...,
        weight_value_function: float = ...,
        weight_entropy: float = ...,
        num_iter: int = ...,
        target_update_frequency: int = ...,
        train_frequency: int = ...,
        num_rollouts: int = ...,
        gamma: float = ...,
        exploration_steps: int = ...,
        exploration_episodes: int = ...,
        tensorboard: bool = ...,
        comment: str = ...,
    ) -> None: ...
    def _early_stop_training(self, *args, **kwargs):
        """Early stop the training algorithm."""
        return kwargs.get("approximate_kl", self.target_kl) >= 1.5 * self.target_kl
