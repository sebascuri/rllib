from typing import Optional, Union

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
        batch_size: int = ...,
        num_iter: int = ...,
        regularization: bool = ...,
        train_frequency: int = ...,
        num_rollouts: int = ...,
        gamma: float = ...,
        exploration_steps: int = ...,
        exploration_episodes: int = ...,
        tensorboard: bool = ...,
        comment: str = ...,
    ) -> None: ...
