from typing import Any, Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution

from rllib.dataset.datatypes import Loss
from rllib.policy import AbstractPolicy
from rllib.util.parameter_decay import ParameterDecay

from .abstract_algorithm import AbstractAlgorithm
from .policy_evaluation.abstract_td_target import AbstractTDTarget

class MPOWorker(nn.Module):
    eta: ParameterDecay
    eta_mean: ParameterDecay
    eta_var: ParameterDecay

    epsilon: Tensor
    epsilon_mean: Tensor
    epsilon_var: Tensor
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
    num_action_samples: int
    mpo_loss: MPOWorker
    ope: Optional[AbstractTDTarget]
    def __init__(
        self,
        num_action_samples: int = ...,
        epsilon: Union[ParameterDecay, float] = ...,
        epsilon_mean: Union[ParameterDecay, float] = ...,
        epsilon_var: Optional[Union[ParameterDecay, float]] = ...,
        regularization: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def get_kl_and_pi(self, state: Tensor) -> Tuple[Tensor, Tensor, Distribution]: ...
    def reset(self) -> None: ...
