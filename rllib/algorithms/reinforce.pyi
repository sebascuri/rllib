from typing import NamedTuple, List

import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss

from rllib.dataset.datatypes import Observation
from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractValueFunction


class REINFORCELoss(NamedTuple):
    actor_loss: Tensor
    baseline_loss: Tensor


class REINFORCE(nn.Module):
    eps: float = 1e-12
    policy: AbstractPolicy
    baseline: AbstractValueFunction
    criterion: _Loss
    gamma: float

    def __init__(self, policy: AbstractPolicy, baseline: AbstractValueFunction,
                 criterion: _Loss, gamma: float) -> None: ...

    def forward(self, *args: List[Observation], **kwargs) -> REINFORCELoss: ...

    def update(self) -> None: ...