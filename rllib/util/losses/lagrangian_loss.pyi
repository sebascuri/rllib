from typing import Any, Union

import torch.nn as nn
from torch import Tensor

from rllib.dataset.datatypes import Loss
from rllib.util.parameter_decay import ParameterDecay

class LagrangianLoss(nn.Module):
    _dual: ParameterDecay
    inequality_zero: Union[float, Tensor]
    regularization: bool
    def __init__(
        self,
        dual: Union[ParameterDecay, float] = ...,
        inequality_zero: Union[float, Tensor] = ...,
        regularization: bool = ...,
    ) -> None: ...
    @property
    def dual(self) -> Tensor: ...
    def forward(self, inequality_value: Tensor, **kwargs: Any) -> Loss: ...
