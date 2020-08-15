from typing import Any

import torch.nn as nn

from rllib.value_function import AbstractValueFunction

from .mpo import MPO

class VMPO(MPO):
    critic: AbstractValueFunction
    critic_target: AbstractValueFunction

    top_k_fraction: float
    def __init__(
        self, top_k_fraction: float = ..., *args: Any, **kwargs: Any,
    ) -> None: ...
