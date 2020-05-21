from abc import ABCMeta
from typing import NamedTuple

import torch.nn as nn
from torch import Tensor


class AbstractAlgorithm(nn.Module, metaclass=ABCMeta):
    """Abstract Algorithm template."""

    def update(self) -> None: ...


class TDLoss(NamedTuple):
    loss: Tensor
    td_error: Tensor


class ACLoss(NamedTuple):
    actor_loss: Tensor
    critic_loss: Tensor
    td_error: Tensor


class PGLoss(NamedTuple):
    actor_loss: Tensor
    baseline_loss: Tensor


# LPLoss = namedtuple('REPSLosses', ['dual', 'primal', 'td', 'advantage'])
