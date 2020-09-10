from typing import Any, Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution

from rllib.dataset.datatypes import Loss
from rllib.policy import AbstractPolicy
from rllib.util.parameter_decay import ParameterDecay

from .abstract_algorithm import AbstractAlgorithm
from .kl_loss import KLLoss
from .policy_evaluation.abstract_td_target import AbstractTDTarget

class MPOLoss(nn.Module):
    eta: ParameterDecay
    epsilon: Tensor
    def __init__(
        self,
        epsilon: Union[ParameterDecay, float] = ...,
        epsilon_mean: Union[ParameterDecay, float] = ...,
        epsilon_var: Optional[Union[ParameterDecay, float]] = ...,
        regularization: bool = ...,
    ) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> Loss: ...

class MPO(AbstractAlgorithm):
    old_policy: AbstractPolicy
    mpo_loss: MPOLoss
    kl_loss: KLLoss
    ope: Optional[AbstractTDTarget]
    def __init__(
        self,
        num_samples: int = ...,
        epsilon: Union[ParameterDecay, float] = ...,
        epsilon_mean: Union[ParameterDecay, float] = ...,
        epsilon_var: Optional[Union[ParameterDecay, float]] = ...,
        regularization: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def get_kl_and_pi(self, state: Tensor) -> Tuple[Tensor, Tensor, Distribution]: ...
    def reset(self) -> None: ...
