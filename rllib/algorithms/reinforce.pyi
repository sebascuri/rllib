from torch import Tensor
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from typing import NamedTuple, List
from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractValueFunction
from rllib.dataset.datatypes import Observation

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

    def _value_estimate(self, trajectory: Observation) -> Tensor: ...

    def forward(self, *args: List[Observation], **kwargs) -> REINFORCELoss: ...
