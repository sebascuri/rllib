"""Actor-Critic Algorithm."""
from typing import List

import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss

from rllib.dataset.datatypes import Observation
from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractQFunction

from .abstract_algorithm import AbstractAlgorithm, ACLoss

class ActorCritic(AbstractAlgorithm):
    policy: AbstractPolicy
    policy_target: AbstractPolicy
    critic: AbstractQFunction
    critic_target: AbstractQFunction
    criterion: _Loss
    gamma: float
    eps: float = 1e-12
    num_samples: int
    def __init__(
        self,
        policy: AbstractPolicy,
        critic: AbstractQFunction,
        criterion: _Loss,
        gamma: float = ...,
        num_samples: int = ...,
    ) -> None: ...
    def returns(self, trajectory: Observation) -> Tensor: ...
    def forward_slow(self, trajectories: List[Observation]) -> ACLoss: ...
    def forward(self, trajectories: List[Observation], **kwargs) -> ACLoss: ...
