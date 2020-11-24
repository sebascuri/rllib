from abc import ABCMeta, abstractmethod
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from rllib.model import AbstractModel
from rllib.value_function import AbstractValueFunction

class MPCSolver(nn.Module, metaclass=ABCMeta):
    dynamical_model: AbstractModel
    reward_model: AbstractModel
    horizon: int
    gamma: float
    num_iter: int
    num_samples: int
    termination_model: Optional[AbstractModel]
    terminal_reward: AbstractValueFunction
    warm_start: bool
    default_action: str
    action_scale: Tensor
    clamp: bool

    mean: Optional[Tensor]
    _scale: float
    covariance: Tensor
    def __init__(
        self,
        dynamical_model: AbstractModel,
        reward_model: AbstractModel,
        horizon: int = ...,
        gamma: float = ...,
        scale: float = ...,
        num_iter: int = ...,
        num_samples: Optional[int] = ...,
        termination_model: Optional[AbstractModel] = ...,
        terminal_reward: Optional[AbstractValueFunction] = ...,
        warm_start: bool = ...,
        default_action: str = ...,
        action_scale: float = ...,
        clamp: bool = ...,
        num_cpu: int = ...,
    ) -> None: ...
    def evaluate_action_sequence(
        self, action_sequence: Tensor, state: Tensor
    ) -> Tensor: ...
    def get_action_sequence_and_returns(self, state: Tensor) -> None: ...
    @abstractmethod
    def get_candidate_action_sequence(self) -> Tensor: ...
    @abstractmethod
    def get_best_action(self, action_sequence: Tensor, returns: Tensor) -> Tensor: ...
    @abstractmethod
    def update_sequence_generation(self, elite_actions: Tensor) -> None: ...
    def initialize_actions(self, batch_shape: torch.Size) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> Tensor: ...
    def reset(self, warm_action: Optional[Tensor] = ...) -> None: ...
