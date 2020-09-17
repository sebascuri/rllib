from typing import Any, Type, Union

from torch.nn.modules.loss import _Loss

from rllib.algorithms.sac import SAC
from rllib.policy import AbstractPolicy
from rllib.util.parameter_decay import ParameterDecay
from rllib.value_function import NNQFunction

from .off_policy_agent import OffPolicyAgent

class SACAgent(OffPolicyAgent):
    algorithm: SAC
    def __init__(
        self,
        critic: NNQFunction,
        policy: AbstractPolicy,
        criterion: Type[_Loss] = ...,
        eta: Union[float, ParameterDecay] = ...,
        regularization: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
