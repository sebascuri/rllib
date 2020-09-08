"""Abstract calculation of TD-Target."""
from abc import ABCMeta, abstractmethod
from typing import Any, Optional

import torch.nn as nn
from torch import Tensor

from rllib.dataset.datatypes import Observation
from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractQFunction, IntegrateQValueFunction

class AbstractTDTarget(nn.Module, metaclass=ABCMeta):
    critic: AbstractQFunction
    policy: Optional[AbstractPolicy]
    gamma: float
    lambda_: float
    num_samples: int
    value_target: IntegrateQValueFunction
    def __init__(
        self,
        critic: AbstractQFunction,
        policy: Optional[AbstractPolicy] = ...,
        gamma: float = ...,
        lambda_: float = ...,
        num_samples: int = ...,
    ) -> None: ...
    @abstractmethod
    def correction(self, pi_log_prob: Tensor, mu_log_prob: Tensor) -> Tensor: ...
    def forward(self, observation: Observation, **kwargs: Any) -> Tensor: ...
    def td(
        self, this_v: Tensor, next_v: Tensor, reward: Tensor, correction: Tensor
    ) -> Tensor: ...
