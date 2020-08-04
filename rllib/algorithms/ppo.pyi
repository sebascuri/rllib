from typing import Any, Tuple, Union

import torch.nn as nn
from torch import Tensor

from rllib.util.parameter_decay import ParameterDecay

from .trpo import TRPO

class PPO(TRPO):
    epsilon: ParameterDecay

    weight_value_function: float
    weight_entropy: float
    clamp_value: bool
    def __init__(
        self,
        epsilon: Union[ParameterDecay, float] = ...,
        weight_value_function: float = ...,
        weight_entropy: float = ...,
        clamp_value: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def get_log_p_and_entropy(
        self, state: Tensor, action: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor,]: ...
