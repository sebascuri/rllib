"""Implementation of MLL loss for multi-output GPs."""
from typing import Union

from torch import Tensor
from torch.distributions import Distribution
import torch.nn as nn
from .gps import ExactGP

def exact_mll(
    predicted_distribution: Distribution,
    target: Tensor,
    gp: Union[nn.ModuleList, ExactGP],
) -> Tensor: ...
