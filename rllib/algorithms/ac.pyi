"""Actor-Critic Algorithm."""
from typing import List

import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss

from .abstract_algorithm import ACLoss, AbstractAlgorithm
from rllib.dataset.datatypes import Observation
from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractQFunction


class ActorCritic(AbstractAlgorithm):
    policy: AbstractPolicy
    policy_target: AbstractPolicy
    critic: AbstractQFunction
    critic_target: AbstractQFunction
    criterion: _Loss
    gamma: float
    eps: float = 1e-12

    def __init__(self, policy: AbstractPolicy, critic: AbstractQFunction,
                 criterion: _Loss, gamma: float) -> None: ...

    def returns(self, trajectory: Observation) -> Tensor: ...

    def forward(self, *args: List[Observation], **kwargs) -> ACLoss: ...
