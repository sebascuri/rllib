from abc import ABCMeta
from typing import Any, List, Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss

from rllib.dataset.datatypes import Loss, Observation
from rllib.policy import AbstractPolicy
from rllib.util.parameter_decay import ParameterDecay
from rllib.util.utilities import RewardTransformer
from rllib.value_function import AbstractQFunction, IntegrateQValueFunction

from .entropy_loss import EntropyLoss
from .kl_loss import KLLoss
from .pathwise_loss import PathwiseLoss
from .policy_evaluation.abstract_td_target import AbstractTDTarget

class AbstractAlgorithm(nn.Module, metaclass=ABCMeta):
    """Abstract Algorithm template."""

    eps: float = ...
    _info: dict
    gamma: float
    reward_transformer: RewardTransformer
    critic: AbstractQFunction
    critic_target: AbstractQFunction
    policy: AbstractPolicy
    policy_target: AbstractPolicy
    old_policy: AbstractPolicy
    criterion: _Loss
    entropy_loss: EntropyLoss
    kl_loss: KLLoss
    pathwise_loss: Optional[PathwiseLoss]
    num_samples: int
    ope: Optional[AbstractTDTarget]
    value_target: IntegrateQValueFunction
    def __init__(
        self,
        gamma: float,
        policy: AbstractPolicy,
        critic: AbstractQFunction,
        eta: Optional[Union[ParameterDecay, float]] = ...,
        entropy_regularization: bool = ...,
        epsilon_mean: Union[ParameterDecay, float] = ...,
        epsilon_var: Optional[Union[ParameterDecay, float]] = ...,
        kl_regularization: bool = ...,
        num_samples: int = ...,
        criterion: _Loss = ...,
        ope: Optional[AbstractTDTarget] = ...,
        reward_transformer: RewardTransformer = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def post_init(self) -> None: ...
    def update(self) -> None: ...
    def reset(self) -> None: ...
    def info(self) -> dict: ...
    def reset_info(
        self, num_trajectories: int = ..., *args: Any, **kwargs: Any
    ) -> None: ...
    def set_policy(self, new_policy: AbstractPolicy) -> None: ...
    def get_reward(self, observation: Observation) -> Tensor: ...
    def get_value_prediction(self, observation: Observation) -> Tensor: ...
    def get_value_target(self, observation: Observation) -> Tensor: ...
    def get_kl_entropy(self, state: Tensor) -> Tuple[Tensor, Tensor, Tensor]: ...
    def get_log_p_and_ope_weight(
        self, state: Tensor, action: Tensor
    ) -> Tuple[Tensor, Tensor]: ...
    def get_ope_weight(
        self, state: Tensor, action: Tensor, log_prob_action: Tensor
    ) -> Tensor: ...
    def score_actor_loss(
        self, observation: Observation, linearized: bool = ...
    ) -> Loss: ...
    def regularization_loss(self, observation: Observation) -> Loss: ...
    def actor_loss(self, observation: Observation) -> Loss: ...
    def critic_loss(self, observation: Observation) -> Loss: ...
    def forward(
        self, observation: Union[Observation, List[Observation]], **kwargs: Any
    ) -> Loss: ...
