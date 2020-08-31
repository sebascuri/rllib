from typing import Any, Type

from torch.nn.modules.loss import _Loss

from rllib.algorithms.esarsa import ESARSA
from rllib.policy import AbstractQFunctionPolicy
from rllib.value_function import AbstractQFunction

from .on_policy_agent import OnPolicyAgent

class ExpectedSARSAAgent(OnPolicyAgent):
    algorithm: ESARSA
    policy: AbstractQFunctionPolicy
    def __init__(
        self,
        critic: AbstractQFunction,
        policy: AbstractQFunctionPolicy,
        criterion: Type[_Loss],
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
