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
    r"""Implementation of DPG algorithm.

    DPG is an off-policy model-free control algorithm.

    The DPG algorithm is an actor-critic algorithm that has a policy that estimates:
    .. math:: a = \pi(s) = \argmax_a Q(s, a)


    Parameters
    ----------
    q_function: AbstractQFunction
        q_function to optimize.
    criterion: _Loss
        Criterion to optimize.
    gamma: float
        discount factor.

    References
    ----------
    Silver, David, et al. (2014) "Deterministic policy gradient algorithms." JMLR.
    Lillicrap et. al. (2016). CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING. ICLR.
    """

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
