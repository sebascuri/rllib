import torch.nn as nn
from rllib.value_function import AbstractQFunction
from typing import NamedTuple
from torch import Tensor
from torch.nn.modules.loss import _Loss
from rllib.policy.q_function_policy import SoftMax


class QLearningLoss(NamedTuple):
    loss: Tensor
    td_error: Tensor


class QLearning(nn.Module):
    q_function: AbstractQFunction
    q_target: AbstractQFunction
    criterion: _Loss
    gamma: float

    def __init__(self, q_function: AbstractQFunction, criterion: _Loss, gamma: float) -> None: ...


    def forward(self, *args: Tensor, **kwargs) -> QLearningLoss: ...

    def _build_return(self, pred_q: Tensor, target_q: Tensor) -> QLearningLoss: ...

    def update(self) -> None: ...


class GradientQLearning(QLearning): ...


class DQN(QLearning): ...


class DDQN(QLearning): ...


class SoftQLearning(QLearning):
    policy: SoftMax
    policy_target: SoftMax

    def __init__(self, q_function: AbstractQFunction, criterion: _Loss,
                 temperature: float, gamma: float) -> None: ...