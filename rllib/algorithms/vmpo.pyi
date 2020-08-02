from typing import Any, Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution
from torch.nn.modules.loss import _Loss

from rllib.dataset.datatypes import Observation
from rllib.policy import AbstractPolicy
from rllib.util.parameter_decay import ParameterDecay
from rllib.util.utilities import RewardTransformer
from rllib.value_function import AbstractValueFunction

from .abstract_algorithm import AbstractAlgorithm, MPOLoss
from .mpo import MPOWorker

class VMPO(AbstractAlgorithm):
    old_policy: AbstractPolicy
    policy: AbstractPolicy
    value_function: AbstractValueFunction
    value_target: AbstractValueFunction

    gamma: float
    top_k_fraction: float

    mpo_loss: MPOWorker
    value_loss: _Loss
    reward_transformer: RewardTransformer
    def __init__(
        self,
        policy: AbstractPolicy,
        value_function: AbstractValueFunction,
        criterion: _Loss,
        epsilon: Union[ParameterDecay, float] = ...,
        epsilon_mean: Union[ParameterDecay, float] = ...,
        epsilon_var: Optional[Union[ParameterDecay, float]] = ...,
        regularization: bool = ...,
        top_k_fraction: float = ...,
        gamma: float = ...,
        reward_transformer: RewardTransformer = ...,
    ) -> None: ...
    def get_kl_and_pi(self, state: Tensor) -> Tuple[Tensor, Tensor, Distribution]: ...
    def get_value_target(
        self, reward: Tensor, next_state: Tensor, done: Tensor
    ) -> Tensor: ...
    def reset(self) -> None: ...
    def forward(self, observation: Observation, **kwargs: Any) -> MPOLoss: ...
