from typing import Any, Type

from torch.nn.modules.loss import _Loss

from rllib.algorithms.q_learning import QLearning
from rllib.policy import AbstractQFunctionPolicy
from rllib.value_function import AbstractQFunction

from .off_policy_agent import OffPolicyAgent

class QLearningAgent(OffPolicyAgent):
    algorithm: QLearning
    policy: AbstractQFunctionPolicy
    def __init__(
        self,
        critic: AbstractQFunction,
        policy: AbstractQFunctionPolicy,
        criterion: Type[_Loss] = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
