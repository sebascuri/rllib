"""KL-Constraint Helper Module."""
from typing import Any, Optional, Union

import torch.nn as nn
from torch import Tensor

from rllib.dataset.datatypes import Loss
from rllib.util.parameter_decay import ParameterDecay

class KLLoss(nn.Module):
    _eta_mean: ParameterDecay
    _eta_var: ParameterDecay
    epsilon_mean: Tensor
    epsilon_var: Tensor
    regularization: bool
    separated_kl: bool
    def __init__(
        self,
        epsilon_mean: Union[ParameterDecay, float] = ...,
        epsilon_var: Optional[Union[ParameterDecay, float]] = ...,
        regularization: bool = ...,
    ) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> Loss: ...
