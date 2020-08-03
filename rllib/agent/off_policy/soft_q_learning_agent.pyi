from typing import Any, Type, Union

from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from rllib.agent import QLearningAgent
from rllib.algorithms.soft_q_learning import SoftQLearning
from rllib.dataset import ExperienceReplay
from rllib.policy import SoftMax
from rllib.util.parameter_decay import ParameterDecay
from rllib.value_function import AbstractQFunction

class SoftQLearningAgent(QLearningAgent):
    algorithm: SoftQLearning
    policy: SoftMax
    def __init__(
        self,
        q_function: AbstractQFunction,
        criterion: Type[_Loss],
        optimizer: Optimizer,
        memory: ExperienceReplay,
        temperature: Union[float, ParameterDecay],
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
