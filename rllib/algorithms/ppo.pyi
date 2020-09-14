from typing import Any, Union

import torch.nn as nn

from rllib.util.parameter_decay import ParameterDecay

from .trpo import TRPO

class PPO(TRPO):
    epsilon: ParameterDecay
    clamp_value: bool
    def __init__(
        self,
        epsilon: Union[ParameterDecay, float] = ...,
        clamp_value: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
