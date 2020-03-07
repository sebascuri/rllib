from torch import Tensor
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from typing import NamedTuple
from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractQFunction


class DPGLoss(NamedTuple):
    actor_loss: Tensor
    critic_loss: Tensor
    td_error: Tensor


class DPG(nn.Module):
    q_function: AbstractQFunction
    q_target: AbstractQFunction
    policy: AbstractPolicy
    policy_target: AbstractPolicy
    criterion: _Loss
    gamma: float
    policy_noise: float
    noise_clip: float

    def __init__(self, q_function: AbstractQFunction, policy: AbstractPolicy, criterion: _Loss,
                 policy_noise: float, noise_clip: float, gamma: float) -> None: ...

    def _add_noise(self, action: Tensor) -> Tensor: ...

    def _actor_loss(self, state: Tensor) -> Tensor: ...

    def forward(self, *args: Tensor, **kwargs) -> DPGLoss: ...

    def update(self) -> None: ...
