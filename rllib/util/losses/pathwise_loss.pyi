from typing import Any, Optional

import torch.nn as nn
from torch import Tensor

from rllib.dataset.datatypes import Loss, Observation
from rllib.policy import AbstractPolicy
from rllib.util.multi_objective_reduction import AbstractMultiObjectiveReduction
from rllib.value_function import AbstractQFunction

class PathwiseLoss(nn.Module):
    policy: Optional[AbstractPolicy]
    critic: Optional[AbstractQFunction]
    multi_objective_reduction: AbstractMultiObjectiveReduction
    def __init__(
        self,
        policy: Optional[AbstractPolicy] = ...,
        critic: Optional[AbstractQFunction] = ...,
        multi_objective_reduction: AbstractMultiObjectiveReduction = ...,
    ) -> None: ...
    def forward(self, observation: Observation, **kwargs: Any) -> Loss: ...

def get_q_value_pathwise_gradients(
    critic: AbstractQFunction,
    state: Tensor,
    action: Tensor,
    multi_objective_reduction: AbstractMultiObjectiveReduction,
) -> Tensor: ...
