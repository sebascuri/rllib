"""Abstract Algorithm."""

from abc import ABCMeta
from collections import namedtuple

import torch.jit
import torch.nn as nn


class AbstractAlgorithm(nn.Module, metaclass=ABCMeta):
    """Abstract Algorithm template."""

    def __init__(self):
        super().__init__()
        self._info = {}

    @torch.jit.export
    def update(self):
        """Update algorithm parameters."""
        pass

    @torch.jit.export
    def reset(self):
        """Reset algorithms parameters."""
        pass

    def info(self):
        """Return info parameters for logging."""
        return self._info


TDLoss = namedtuple("TDLoss", ["loss", "td_error"])
ACLoss = namedtuple("ACLoss", ["loss", "policy_loss", "critic_loss", "td_error"])
PGLoss = namedtuple("PGLoss", ["loss", "policy_loss", "baseline_loss"])
LPLoss = namedtuple("LPLoss", ["loss", "dual", "policy_loss"])
MPOLoss = namedtuple(
    "MPOLoss", ["loss", "dual", "policy_loss", "critic_loss", "td_error"]
)
SACLoss = namedtuple(
    "SACLoss", ["loss", "policy_loss", "critic_loss", "eta_loss", "td_error"]
)
PPOLoss = namedtuple("PPOLoss", ["loss", "surrogate_loss", "critic_loss", "entropy"])
