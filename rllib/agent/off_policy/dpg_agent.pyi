from typing import Any, Type, Union

from torch.nn.modules.loss import _Loss

from rllib.algorithms.dpg import DPG
from rllib.policy import AbstractPolicy
from rllib.util.parameter_decay import ParameterDecay
from rllib.value_function import AbstractQFunction

from .off_policy_agent import OffPolicyAgent

class DPGAgent(OffPolicyAgent):
    algorithm: DPG
    def __init__(
        self,
        critic: AbstractQFunction,
        policy: AbstractPolicy,
        exploration_noise: Union[float, ParameterDecay] = ...,
        criterion: Type[_Loss] = ...,
        policy_noise: float = ...,
        noise_clip: float = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
