"""KL-Constraint Helper Module."""
from typing import Any, Union, Optional
from torch import Tensor
import torch.nn as nn

from rllib.dataset.datatypes import Loss
from rllib.util.parameter_decay import ParameterDecay

class KLLoss(nn.Module):
    eta_mean: ParameterDecay
    eta_var: ParameterDecay
    epsilon_mean: Tensor
    epsilon_var: Tensor
    regularization: bool
    def __init__(
        self,
        epsilon_mean: Union[ParameterDecay, float] = ...,
        epsilon_var: Optional[Union[ParameterDecay, float]] = ...,
        regularization: bool = ...,
    ) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> Loss: ...
