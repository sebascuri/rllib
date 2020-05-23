from abc import ABCMeta
from typing import NamedTuple

import torch.nn as nn
from torch import Tensor


class AbstractAlgorithm(nn.Module, metaclass=ABCMeta):
    """Abstract Algorithm template."""

    _info: dict

    def update(self) -> None: ...

    def reset(self) -> None: ...

    def info(self) -> dict: ...


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
