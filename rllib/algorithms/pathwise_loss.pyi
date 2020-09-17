from typing import Any, Optional

import torch.nn as nn

from rllib.dataset.datatypes import Loss, Observation
from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractQFunction

class PathwiseLoss(nn.Module):
    policy: Optional[AbstractPolicy]
    critic: Optional[AbstractQFunction]
    def __init__(
        self, policy: Optional[AbstractPolicy], critic: Optional[AbstractQFunction]
    ) -> None: ...
    def forward(self, observation: Observation, **kwargs: Any) -> Loss: ...
