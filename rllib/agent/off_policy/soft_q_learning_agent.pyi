from typing import Any, Union

from rllib.agent import QLearningAgent
from rllib.algorithms.soft_q_learning import SoftQLearning
from rllib.policy import SoftMax
from rllib.util.parameter_decay import ParameterDecay
from rllib.value_function import AbstractQFunction

class SoftQLearningAgent(QLearningAgent):
    algorithm: SoftQLearning
    policy: SoftMax
    def __init__(
        self,
        critic: AbstractQFunction,
        temperature: Union[float, ParameterDecay] = 0.2,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
