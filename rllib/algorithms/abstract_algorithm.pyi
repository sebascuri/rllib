from abc import ABCMeta
from typing import Any, List, Union

import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss

from rllib.dataset.datatypes import Loss, Observation
from rllib.policy import AbstractPolicy
from rllib.util.utilities import RewardTransformer
from rllib.value_function import AbstractQFunction, IntegrateQValueFunction

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
    criterion: _Loss
    entropy_regularization: float
    num_samples: int
    value_target: IntegrateQValueFunction
    def __init__(
        self,
        gamma: float,
        policy: AbstractPolicy,
        critic: AbstractQFunction,
        entropy_regularization: float = ...,
        num_samples: int = ...,
        criterion: _Loss = ...,
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
    def get_value_target(self, observation: Observation) -> Tensor: ...
    def process_value_prediction(
        self, value_prediction: Tensor, observation: Observation
    ) -> Tensor: ...
    def actor_loss(self, observation: Observation) -> Loss: ...
    def critic_loss(self, observation: Observation) -> Loss: ...
    def forward(
        self, observation: Union[Observation, List[Observation]], **kwargs: Any
    ) -> Loss: ...
