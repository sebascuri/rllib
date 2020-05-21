"""Abstract Algorithm."""

from abc import ABCMeta
from collections import namedtuple

import torch.nn as nn
import torch.jit


class AbstractAlgorithm(nn.Module, metaclass=ABCMeta):
    """Abstract Algorithm template."""

    @torch.jit.export
    def update(self):
        """Update algorithm parameters."""
        pass


TDLoss = namedtuple('TDLoss', ['loss', 'td_error'])
ACLoss = namedtuple('ACLoss', ['actor_loss', 'critic_loss', 'td_error'])
PGLoss = namedtuple('PGLoss', ['actor_loss', 'baseline_loss'])
# LPLoss = namedtuple('REPSLosses', ['dual', 'primal', 'td', 'advantage'])
