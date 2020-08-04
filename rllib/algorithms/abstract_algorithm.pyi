from abc import ABCMeta
from typing import Any, NamedTuple

import torch.nn as nn
from torch import Tensor

from rllib.dataset.datatypes import Observation
from rllib.policy import AbstractPolicy
from rllib.util.utilities import RewardTransformer

class AbstractAlgorithm(nn.Module, metaclass=ABCMeta):
    """Abstract Algorithm template."""

    eps: float = ...
    _info: dict
    gamma: float
    reward_transformer: RewardTransformer
    policy: AbstractPolicy
    def __init__(
        self,
        gamma: float,
        policy: AbstractPolicy,
        reward_transformer: RewardTransformer = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def update(self) -> None: ...
    def reset(self) -> None: ...
    def info(self) -> dict: ...
    def get_q_target(self, observation: Observation) -> Tensor: ...

class TDLoss(NamedTuple):
    loss: Tensor
    td_error: Tensor

class ACLoss(NamedTuple):
    loss: Tensor
    policy_loss: Tensor
    critic_loss: Tensor
    td_error: Tensor

class PGLoss(NamedTuple):
    loss: Tensor
    policy_loss: Tensor
    baseline_loss: Tensor

class LPLoss(NamedTuple):
    loss: Tensor
    dual: Tensor
    policy_loss: Tensor

class MPOLoss(NamedTuple):
    loss: Tensor
    dual: Tensor
    policy_loss: Tensor
    critic_loss: Tensor
    td_error: Tensor

class SACLoss(NamedTuple):
    loss: Tensor
    policy_loss: Tensor
    critic_loss: Tensor
    eta_loss: Tensor
    td_error: Tensor

class PPOLoss(NamedTuple):
    loss: Tensor
    surrogate_loss: Tensor
    critic_loss: Tensor
    entropy: Tensor

class TRPOLoss(NamedTuple):
    loss: Tensor
    surrogate_loss: Tensor
    critic_loss: Tensor
    dual_loss: Tensor
    kl_loss: Tensor
