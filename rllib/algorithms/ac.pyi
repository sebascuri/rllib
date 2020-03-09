"""Actor-Critic Algorithm."""
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.loss import _Loss

from typing import NamedTuple, List
from rllib.dataset.datatypes import Observation
from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractQFunction, AbstractValueFunction


class PGLoss(NamedTuple):
    actor_loss: Tensor
    critic_loss: Tensor

class ActorCritic(nn.Module):
    policy: AbstractPolicy
    policy_target: AbstractPolicy
    critic: AbstractValueFunction
    critic_target: AbstractValueFunction
    criterion: _Loss
    gamma: float
    eps: float = 1e-12

    def __init__(self, policy: AbstractPolicy, critic: AbstractQFunction,
                 criterion: _Loss, gamma: float) -> None: ...

    def returns(self, trajectory: Observation) -> Tensor: ...

    def forward(self, *args: List[Observation], **kwargs) -> PGLoss: ...

    def update(self) -> None: ...
