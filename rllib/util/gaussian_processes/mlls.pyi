"""Implementation of MLL loss for multi-output GPs."""
from typing import Union

import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution

from .gps import ExactGP

def exact_mll(
    predicted_distribution: Distribution,
    target: Tensor,
    gp: Union[nn.ModuleList, ExactGP],
) -> Tensor: ...
