from typing import Any

from torch import Tensor
from torch.nn.modules.loss import _Loss

from rllib.dataset.datatypes import Observation
from rllib.value_function import AbstractQFunction

from .abstract_algorithm import AbstractAlgorithm, TDLoss

class QLearning(AbstractAlgorithm):
    q_function: AbstractQFunction
    q_target: AbstractQFunction
    criterion: _Loss
    gamma: float
    def __init__(
        self, q_function: AbstractQFunction, criterion: _Loss, gamma: float
    ) -> None: ...
    def get_q_target(
        self, reward: Tensor, next_state: Tensor, done: Tensor
    ) -> Tensor: ...
    def forward(self, observation: Observation, **kwargs: Any) -> TDLoss: ...
    def _build_return(self, pred_q: Tensor, target_q: Tensor) -> TDLoss: ...
    def update(self) -> None: ...

class GradientQLearning(QLearning): ...
