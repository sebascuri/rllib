import torch.nn as nn
from rllib.value_function import AbstractQFunction
from rllib.policy import AbstractPolicy
from .q_learning import QLearningLoss
from torch import Tensor
from torch.nn.modules.loss import _Loss


class ESARSA(nn.Module):
    q_function: AbstractQFunction
    q_target: AbstractQFunction
    policy: AbstractPolicy
    criterion: _Loss
    gamma: float

    def __init__(self, q_function: AbstractQFunction, criterion: _Loss,
                 policy: AbstractPolicy, gamma: float) -> None: ...

    def forward(self, *args: Tensor, **kwargs) -> QLearningLoss: ...

    def _build_return(self, pred_q: Tensor, target_q: Tensor) -> QLearningLoss: ...

    def update(self) -> None: ...

class GradientExpectedSARSA(ESARSA): ...
