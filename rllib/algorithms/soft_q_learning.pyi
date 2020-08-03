from typing import Union

from torch.nn.modules.loss import _Loss

from rllib.policy.q_function_policy import SoftMax
from rllib.util.parameter_decay import ParameterDecay
from rllib.value_function import AbstractQFunction

from .q_learning import QLearning

class SoftQLearning(QLearning):
    policy: SoftMax
    policy_target: SoftMax
    def __init__(
        self,
        q_function: AbstractQFunction,
        criterion: _Loss,
        temperature: Union[float, ParameterDecay],
        gamma: float,
    ) -> None: ...
