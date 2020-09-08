from typing import Any

from rllib.algorithms.policy_evaluation.gae import GAE
from rllib.value_function import AbstractValueFunction

from .ac import ActorCritic

class GAAC(ActorCritic):
    gae: GAE
    critic: AbstractValueFunction
    critic_target: AbstractValueFunction
    def __init__(self, lambda_: float, *args: Any, **kwargs: Any) -> None: ...
