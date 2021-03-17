from typing import Any, Optional, Union

import torch.nn as nn
from torch import Tensor

from rllib.dataset.datatypes import Action, Loss, State
from rllib.util.parameter_decay import ParameterDecay

from .abstract_algorithm import AbstractAlgorithm

class MPOLoss(nn.Module):
    _eta: ParameterDecay
    epsilon: Tensor
    def __init__(
        self,
        epsilon: Union[ParameterDecay, float] = ...,
        epsilon_mean: Union[ParameterDecay, float] = ...,
        epsilon_var: Optional[Union[ParameterDecay, float]] = ...,
        kl_regularization: bool = ...,
    ) -> None: ...
    @property
    def eta(self) -> Tensor: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> Loss: ...

class MPO(AbstractAlgorithm):
    mpo_loss: MPOLoss
    def __init__(
        self,
        num_samples: int = ...,
        epsilon: Union[ParameterDecay, float] = ...,
        epsilon_mean: Union[ParameterDecay, float] = ...,
        epsilon_var: Optional[Union[ParameterDecay, float]] = ...,
        kl_regularization: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def compute_mpo_loss(self, state: State, action: Action) -> Loss: ...
    def reset(self) -> None: ...
