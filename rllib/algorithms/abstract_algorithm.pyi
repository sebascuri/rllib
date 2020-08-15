from abc import ABCMeta
from dataclasses import dataclass
from typing import Any, List, NamedTuple, Union

import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss

from rllib.dataset.datatypes import Observation
from rllib.policy import AbstractPolicy
from rllib.util.utilities import RewardTransformer
from rllib.value_function.abstract_value_function import AbstractQFunction

class AbstractAlgorithm(nn.Module, metaclass=ABCMeta):
    """Abstract Algorithm template."""

    eps: float = ...
    _info: dict
    gamma: float
    reward_transformer: RewardTransformer
    critic: AbstractQFunction
    policy: AbstractPolicy
    criterion: _Loss
    def __init__(
        self,
        gamma: float,
        policy: AbstractPolicy,
        critic: AbstractQFunction,
        criterion: _Loss = ...,
        reward_transformer: RewardTransformer = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def update(self) -> None: ...
    def reset(self) -> None: ...
    def info(self) -> dict: ...
    def get_value_target(self, observation: Observation) -> Tensor: ...
    def process_value_prediction(
        self, value_prediction: Tensor, observation: Observation
    ) -> Tensor: ...
    def actor_loss(self, observation: Observation) -> Loss: ...
    def critic_loss(self, observation: Observation) -> Loss: ...
    def forward_slow(
        self, observation: Union[Observation, List[Observation]]
    ) -> Loss: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> Loss: ...

@dataclass
class Loss:
    loss: Tensor
    td_error: Tensor = ...
    policy_loss: Tensor = ...
    critic_loss: Tensor = ...
    regularization_loss: Tensor = ...
    dual_loss: Tensor = ...
