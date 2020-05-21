import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss

from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractQFunction
from .abstract_algorithm import AbstractAlgorithm, ACLoss, TDLoss


class DPG(AbstractAlgorithm):
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

    def actor_loss(self, state: Tensor) -> Tensor: ...

    def critic_loss(self, state: Tensor, action: Tensor, reward: Tensor,
                    next_state: Tensor, done: Tensor) -> TDLoss: ...

    def forward(self, *args: Tensor, **kwargs) -> ACLoss: ...
