"""Generalized Actor-Critic Algorithm."""
from .ac import ActorCritic
from .gae import GAE
from rllib.value_function import AbstractValueFunction
from rllib.policy import AbstractPolicy
from torch.nn.modules.loss import _Loss

class GAAC(ActorCritic):
    gae: GAE

    def __init__(self, policy: AbstractPolicy, critic: AbstractValueFunction,
                 criterion: _Loss, lambda_: float, gamma: float) -> None: ...

