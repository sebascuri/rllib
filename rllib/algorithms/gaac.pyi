from typing import Any

from rllib.algorithms.policy_evaluation.gae import GAE

from .ac import ActorCritic

class GAAC(ActorCritic):
    gae: GAE
    def __init__(self, lambda_: float, *args: Any, **kwargs: Any) -> None: ...
