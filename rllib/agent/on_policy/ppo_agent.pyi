from typing import Any

from rllib.algorithms.ppo import PPO
from rllib.dataset.datatypes import Loss

from .actor_critic_agent import ActorCriticAgent

class PPOAgent(ActorCriticAgent):

    algorithm: PPO
    target_kl: float
    def __init__(
        self,
        epsilon: float = ...,
        lambda_: float = ...,
        target_kl: float = ...,
        eta: float = ...,
        monte_carlo_target: bool = ...,
        clamp_value: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def early_stop(self, losses: Loss, **kwargs: Any) -> bool: ...
