from typing import Any, Type

from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from rllib.algorithms.q_learning import QLearning
from rllib.dataset import ExperienceReplay
from rllib.policy import AbstractQFunctionPolicy
from rllib.value_function import AbstractQFunction

from .off_policy_agent import OffPolicyAgent

class QLearningAgent(OffPolicyAgent):
    algorithm: QLearning
    policy: AbstractQFunctionPolicy
    def __init__(
        self,
        q_function: AbstractQFunction,
        policy: AbstractQFunctionPolicy,
        criterion: Type[_Loss],
        optimizer: Optimizer,
        memory: ExperienceReplay,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
