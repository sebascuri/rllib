from typing import Any, Optional, Type

from torch.nn.modules.loss import _Loss

from rllib.algorithms.reinforce import REINFORCE
from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractValueFunction

from .on_policy_agent import OnPolicyAgent

class REINFORCEAgent(OnPolicyAgent):
    algorithm: REINFORCE
    def __init__(
        self,
        policy: AbstractPolicy,
        critic: Optional[AbstractValueFunction] = ...,
        criterion: Optional[Type[_Loss]] = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
