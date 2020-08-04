from typing import Any, NamedTuple, Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution
from torch.nn.modules.loss import _Loss

from rllib.dataset.datatypes import Observation
from rllib.policy import AbstractPolicy
from rllib.util.parameter_decay import ParameterDecay
from rllib.value_function import AbstractQFunction

from .abstract_algorithm import AbstractAlgorithm, MPOLoss

class MPOLosses(NamedTuple):
    primal_loss: Tensor
    dual_loss: Tensor

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
    def forward(self, *args: Tensor, **kwargs: Any) -> MPOLosses: ...

class MPO(AbstractAlgorithm):
    old_policy: AbstractPolicy
    q_function: AbstractQFunction
    q_target: AbstractQFunction

    num_action_samples: int

    mpo_loss: MPOWorker
    value_loss: _Loss
    def __init__(
        self,
        q_function: AbstractQFunction,
        num_action_samples: int,
        criterion: _Loss,
        epsilon: Union[ParameterDecay, float] = ...,
        epsilon_mean: Union[ParameterDecay, float] = ...,
        epsilon_var: Optional[Union[ParameterDecay, float]] = ...,
        regularization: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def get_kl_and_pi(self, state: Tensor) -> Tuple[Tensor, Tensor, Distribution]: ...
    def get_q_target(
        self, reward: Tensor, next_state: Tensor, done: Tensor
    ) -> Tensor: ...
    def reset(self) -> None: ...
    def forward(self, observation: Observation, **kwargs: Any) -> MPOLoss: ...