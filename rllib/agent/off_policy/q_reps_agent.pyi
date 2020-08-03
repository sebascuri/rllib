from typing import Any, Optional, Union

from torch.optim.optimizer import Optimizer

from rllib.algorithms.reps import QREPS
from rllib.dataset import ExperienceReplay
from rllib.policy import AbstractPolicy
from rllib.util.parameter_decay import ParameterDecay
from rllib.value_function import AbstractQFunction, AbstractValueFunction

from .reps_agent import REPSAgent

class QREPSAgent(REPSAgent):
    algorithm: QREPS
    def __init__(
        self,
        policy: Optional[AbstractPolicy],
        q_function: AbstractQFunction,
        value_function: AbstractValueFunction,
        optimizer: Optimizer,
        memory: ExperienceReplay,
        epsilon: Union[float, ParameterDecay],
        regularization: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
