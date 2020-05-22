"""Soft Actor-Critic Algorithm."""

from .abstract_algorithm import AbstractAlgorithm, ACLoss, TDLoss
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss

from rllib.dataset.datatypes import Termination
from rllib.model import AbstractModel
from rllib.policy import AbstractPolicy
from rllib.reward import AbstractReward
from rllib.value_function import AbstractQFunction


class SoftActorCritic(AbstractAlgorithm):
    q_function: AbstractQFunction
    q_target: AbstractQFunction
    policy: AbstractPolicy
    policy_target: AbstractPolicy
    criterion: _Loss
    gamma: float

    temperature: float

    def __init__(self, policy: AbstractPolicy, q_function: AbstractQFunction,
                 criterion: _Loss, temperature: float, gamma: float) -> None: ...

    def get_q_target(self, reward: Tensor, next_state: Tensor, done: Tensor
                     ) -> Tensor: ...

    def actor_loss(self, state: Tensor) -> Tensor: ...

    def critic_loss(self, state: Tensor, action: Tensor, q_target: Tensor
                    ) -> TDLoss: ...

    def forward(self, *args: Tensor, **kwargs) -> ACLoss: ...


class MBSoftActorCritic(SoftActorCritic):
    """Model Based Soft-Actor Critic."""
    dynamical_model: AbstractModel
    reward_model: AbstractReward
    termination: Termination

    num_steps: int
    num_samples: int

    def __init__(self, policy: AbstractPolicy, q_function: AbstractQFunction,
                 dynamical_model: AbstractModel, reward_model: AbstractReward,
                 criterion: _Loss, temperature: float, gamma: float,
                 termination: Termination=None, num_steps: int = 1, num_samples: int = 15,
                 ) -> None: ...
