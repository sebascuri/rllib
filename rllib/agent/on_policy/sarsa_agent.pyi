from typing import Any, Type

from torch.nn.modules.loss import _Loss

from rllib.algorithms.sarsa import SARSA
from rllib.policy import AbstractQFunctionPolicy
from rllib.value_function import AbstractQFunction

from .on_policy_agent import OnPolicyAgent

class SARSAAgent(OnPolicyAgent):
    algorithm: SARSA
    policy: AbstractQFunctionPolicy
    def __init__(
        self,
        critic: AbstractQFunction,
        policy: AbstractQFunctionPolicy,
        criterion: Type[_Loss],
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
