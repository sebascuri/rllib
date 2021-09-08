from typing import Any, Union

import torch.nn as nn
from torch import Tensor

from rllib.dataset.datatypes import Loss
from rllib.util.parameter_decay import ParameterDecay

class EntropyLoss(nn.Module):
    _eta: ParameterDecay
    target_entropy: Union[float, Tensor]
    regularization: bool
    def __init__(
        self,
        eta: Union[ParameterDecay, float] = ...,
        target_entropy: Union[float, Tensor] = ...,
        regularization: bool = ...,
    ) -> None: ...
    @property
    def eta(self) -> Tensor: ...
    def forward(self, entropy: Tensor, **kwargs: Any) -> Loss: ...
