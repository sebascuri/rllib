from typing import Union

from torch import Tensor
from torch.nn.modules.loss import _Loss

from .abstract_algorithm import TDLoss, AbstractAlgorithm
from rllib.policy.q_function_policy import SoftMax
from rllib.util.parameter_decay import ParameterDecay
from rllib.value_function import AbstractQFunction



class QLearning(AbstractAlgorithm):
    q_function: AbstractQFunction
    q_target: AbstractQFunction
    criterion: _Loss
    gamma: float

    def __init__(self, q_function: AbstractQFunction, criterion: _Loss, gamma: float) -> None: ...


    def forward(self, *args: Tensor, **kwargs) -> TDLoss: ...

    def _build_return(self, pred_q: Tensor, target_q: Tensor) -> TDLoss: ...

    def update(self) -> None: ...


class GradientQLearning(QLearning): ...


class DQN(QLearning): ...


class DDQN(QLearning): ...


class SoftQLearning(QLearning):
    policy: SoftMax
    policy_target: SoftMax

    def __init__(self, q_function: AbstractQFunction, criterion: _Loss,
                 temperature: Union[float, ParameterDecay], gamma: float) -> None: ...