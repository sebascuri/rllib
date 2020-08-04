from typing import Any

import torch.nn as nn

from rllib.dataset.datatypes import Observation
from rllib.value_function import AbstractValueFunction

from .abstract_algorithm import MPOLoss
from .mpo import MPO

class VMPO(MPO):
    critic: AbstractValueFunction
    critic_target: AbstractValueFunction

    top_k_fraction: float
    def __init__(
        self,
        critic: AbstractValueFunction,
        top_k_fraction: float = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def forward(self, observation: Observation, **kwargs: Any) -> MPOLoss: ...
