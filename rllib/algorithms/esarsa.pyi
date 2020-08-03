from typing import Any

import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss

from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractQFunction

from .abstract_algorithm import AbstractAlgorithm, TDLoss

class ESARSA(AbstractAlgorithm):
    q_function: AbstractQFunction
    q_target: AbstractQFunction
    policy: AbstractPolicy
    criterion: _Loss
    gamma: float
    num_samples: int
    def __init__(
        self,
        q_function: AbstractQFunction,
        criterion: _Loss,
        policy: AbstractPolicy,
        gamma: float = ...,
        num_samples: int = ...,
    ) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> TDLoss: ...
    def _build_return(self, pred_q: Tensor, target_q: Tensor) -> TDLoss: ...
    def update(self) -> None: ...
    def get_q_target(
        self, reward: Tensor, next_state: Tensor, done: Tensor
    ) -> Tensor: ...

class GradientExpectedSARSA(ESARSA): ...
