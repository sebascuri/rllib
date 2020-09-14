from typing import Any

from rllib.algorithms.ppo import PPO
from rllib.dataset.datatypes import Loss
from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractValueFunction

from .actor_critic_agent import ActorCriticAgent

class PPOAgent(ActorCriticAgent):

    algorithm: PPO
    target_kl: float
    def __init__(
        self,
        policy: AbstractPolicy,
        critic: AbstractValueFunction,
        epsilon: float = ...,
        lambda_: float = ...,
        target_kl: float = ...,
        entropy_regularization: float = ...,
        monte_carlo_target: bool = ...,
        clamp_value: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def early_stop(self, losses: Loss, **kwargs: Any) -> bool: ...
