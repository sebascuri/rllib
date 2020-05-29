"""Generalized Actor-Critic Algorithm."""
from torch.nn.modules.loss import _Loss

from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractQFunction

from .ac import ActorCritic
from .gae import GAE


class GAAC(ActorCritic):
    gae: GAE

    def __init__(self, policy: AbstractPolicy, critic: AbstractQFunction,
                 criterion: _Loss, lambda_: float, gamma: float) -> None: ...
