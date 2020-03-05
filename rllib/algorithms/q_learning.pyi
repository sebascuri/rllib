import torch.nn as nn
import copy
from rllib.value_function import AbstractQFunction
from typing import NamedTuple
from torch import Tensor
from torch.nn.modules.loss import _Loss


class QLearningLoss(NamedTuple):
    loss: Tensor
    td_error: Tensor


class QLearning(nn.Module):
    q_function: AbstractQFunction
    criterion: _Loss
    gamma: float
    def __init__(self, q_function: AbstractQFunction, criterion: _Loss, gamma: float) -> None: ...


    def forward(self, *args: Tensor, **kwargs) -> QLearningLoss: ...

    def _build_return(self, pred_q: Tensor, target_q: Tensor) -> QLearningLoss: ...

    def update(self) -> None: ...


class SemiGQLearning(QLearning): ...


class DQN(QLearning): ...


class DDQN(QLearning): ...