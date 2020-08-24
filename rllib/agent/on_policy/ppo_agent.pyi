from typing import Any, Type

from torch.nn.modules.loss import _Loss

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
        criterion: Type[_Loss],
        epsilon: float = ...,
        lambda_: float = ...,
        target_kl: float = ...,
        weight_value_function: float = ...,
        weight_entropy: float = ...,
        monte_carlo_target: bool = ...,
        clamp_value: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def early_stop(self, *args: Any, **kwargs: Any) -> bool:
        """Early stop the training algorithm."""
        return kwargs.get("approximate_kl", self.target_kl) >= 1.5 * self.target_kl
