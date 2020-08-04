from typing import Any

import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss

from rllib.dataset.datatypes import Observation
from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractQFunction

from .abstract_algorithm import AbstractAlgorithm, ACLoss, TDLoss

class DPG(AbstractAlgorithm):
    q_function: AbstractQFunction
    q_target: AbstractQFunction
    policy_target: AbstractPolicy
    criterion: _Loss
    policy_noise: float
    noise_clip: float
    def __init__(
        self,
        q_function: AbstractQFunction,
        criterion: _Loss,
        policy_noise: float,
        noise_clip: float,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def actor_loss(self, state: Tensor) -> Tensor: ...
    def critic_loss(
        self, state: Tensor, action: Tensor, q_target: Tensor
    ) -> TDLoss: ...
    def forward(self, observation: Observation, **kwargs: Any) -> ACLoss: ...
