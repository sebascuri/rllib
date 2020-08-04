from typing import Any, Tuple, Union

import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss

from rllib.dataset.datatypes import Observation
from rllib.util.parameter_decay import ParameterDecay
from rllib.value_function import AbstractQFunction

from .abstract_algorithm import AbstractAlgorithm, SACLoss, TDLoss

class SoftActorCritic(AbstractAlgorithm):
    q_function: AbstractQFunction
    q_target: AbstractQFunction
    criterion: _Loss
    eta: ParameterDecay
    target_entropy: Union[float, Tensor]
    dist_params: dict
    def __init__(
        self,
        q_function: AbstractQFunction,
        criterion: _Loss,
        eta: Union[ParameterDecay, float] = ...,
        regularization: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def actor_loss(self, state: Tensor) -> Tuple[Tensor, Tensor]: ...
    def critic_loss(
        self, state: Tensor, action: Tensor, q_target: Tensor
    ) -> TDLoss: ...
    def forward(self, observation: Observation, **kwargs: Any) -> SACLoss: ...
