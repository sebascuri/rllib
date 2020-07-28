from typing import Type, Union

import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss

from rllib.algorithms.gae import GAE
from rllib.policy import AbstractPolicy
from rllib.util.parameter_decay import ParameterDecay
from rllib.value_function import AbstractValueFunction

from .abstract_algorithm import AbstractAlgorithm, PPOLoss

class PPO(AbstractAlgorithm):
    old_policy: AbstractPolicy
    policy: AbstractPolicy
    value_function: AbstractValueFunction
    value_function_target: AbstractValueFunction
    gamma: float

    epsilon: ParameterDecay

    value_loss_criterion: _Loss
    weight_value_function: float
    weight_entropy: float
    gae: GAE
    eps: float
    def __init__(
        self,
        policy: AbstractPolicy,
        value_function: AbstractValueFunction,
        criterion: Type[_Loss] = ...,
        epsilon: Union[ParameterDecay, float] = ...,
        weight_value_function: float = ...,
        weight_entropy: float = ...,
        lambda_: float = ...,
        gamma: float = ...,
    ) -> None: ...
    def forward(self, *trajectories: Tensor, **kwargs) -> PPOLoss: ...
