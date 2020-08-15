from typing import Any, Union

import torch.nn as nn

from rllib.util.parameter_decay import ParameterDecay

from .gaac import GAAC

class PPO(GAAC):
    epsilon: ParameterDecay
    clamp_value: bool
    monte_carlo_target: bool
    def __init__(
        self,
        epsilon: Union[ParameterDecay, float] = ...,
        clamp_value: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
