from typing import Any, Union

from rllib.policy.q_function_policy import SoftMax
from rllib.util.parameter_decay import ParameterDecay

from .q_learning import QLearning

class SoftQLearning(QLearning):
    policy: SoftMax
    policy_target: SoftMax
    def __init__(
        self, temperature: Union[float, ParameterDecay], *args: Any, **kwargs: Any
    ) -> None: ...
