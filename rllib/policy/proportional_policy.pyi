"""Proportional policy implementation."""
from typing import Any, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from rllib.policy.nn_policy import NNPolicy

class ProportionalModule(nn.Module):
    w: nn.Linear
    def __init__(self, gain: torch.Tensor, fixed: bool = ...) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> Any: ...

class ProportionalPolicy(NNPolicy):
    def __init__(
        self,
        gain: Union[np.ndarray, Tensor, float],
        fixed: bool = ...,
        deterministic: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
