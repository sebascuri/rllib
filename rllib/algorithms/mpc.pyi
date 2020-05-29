"""MPC Algorithms."""
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from rllib.dataset.datatypes import Termination
from rllib.model import AbstractModel
from rllib.reward import AbstractReward
from rllib.util.parameter_decay import ParameterDecay
from rllib.value_function import AbstractValueFunction


class MPCSolver(nn.Module, metaclass=ABCMeta):
    dynamical_model: AbstractModel
    reward_model: AbstractReward
    horizon: int
    gamma: float
    num_iter: int
    num_samples: int
    termination: Optional[Termination]
    terminal_reward: AbstractValueFunction
    warm_start: bool
    default_action: str
    action_scale: Tensor

    mean: Optional[Tensor]
    _scale: float
    covariance: Tensor

    def __init__(self, dynamical_model: AbstractModel, reward_model: AbstractReward,
                 horizon: int, gamma: float = 1., scale: float = 0.3,
                 num_iter: int = 1, num_samples: int = None,
                 termination: Termination = None,
                 terminal_reward: AbstractValueFunction = None,
                 warm_start: bool = False,
                 default_action: str = 'zero',
                 action_scale: float = 1.,
                 num_cpu: int = 1) -> None: ...

    def evaluate_action_sequence(self, action_sequence: Tensor,
                                 state: Tensor) -> Tensor: ...

    def get_action_sequence_and_returns(self, state: Tensor) -> None: ...

    @abstractmethod
    def get_candidate_action_sequence(self) -> Tensor: ...

    @abstractmethod
    def get_best_action(self, action_sequence: Tensor, returns: Tensor) -> Tensor: ...

    @abstractmethod
    def update_sequence_generation(self, elite_actions: Tensor) -> None: ...

    def initialize_actions(self, batch_shape: torch.Size) -> None: ...

    def forward(self, *args: Tensor, **kwargs) -> Tensor: ...

    def reset(self, warm_action: Tensor = None) -> None: ...


class CEMShooting(MPCSolver):
    num_elites: int
    alpha: float

    def __init__(self, dynamical_model: AbstractModel, reward_model: AbstractReward,
                 horizon: int, gamma: float = 1., scale: float = 0.3,
                 num_iter: int = 1, num_samples: int = None, num_elites: int = None,
                 alpha: float = 0., termination: Termination = None,
                 terminal_reward: AbstractValueFunction = None,
                 warm_start: bool = False,
                 action_scale: float = 1.,
                 default_action: str = 'zero', num_cpu: int = 1) -> None: ...

    def get_candidate_action_sequence(self) -> Tensor: ...

    def get_best_action(self, action_sequence: Tensor, returns: Tensor) -> Tensor: ...

    def update_sequence_generation(self, elite_actions: Tensor) -> None: ...


class RandomShooting(CEMShooting):

    def __init__(self, dynamical_model: AbstractModel, reward_model: AbstractReward,
                 horizon: int, gamma: float = 1., scale: float = 0.3,
                 num_samples: int = None, num_elites: int = None,
                 termination: Termination = None,
                 terminal_reward: AbstractValueFunction = None,
                 warm_start: bool = False,
                 action_scale: float = 1.,
                 default_action: str = 'zero', num_cpu: int = 1) -> None: ...


class MPPIShooting(MPCSolver):
    kappa: ParameterDecay
    filter_coefficients: Tensor

    def __init__(self, dynamical_model: AbstractModel, reward_model: AbstractReward,
                 horizon: int, gamma: float = 1., scale: float = 0.3,
                 num_iter: int = 1, num_samples: int = None,
                 kappa: Union[float, ParameterDecay] = 1.,
                 filter_coefficients: List[float] = [1.],
                 termination: Termination = None,
                 terminal_reward: AbstractValueFunction = None,
                 warm_start: bool = False,
                 action_scale: float = 1.,
                 default_action: str = 'zero', num_cpu: int = 1) -> None: ...

    def get_candidate_action_sequence(self) -> Tensor: ...

    def get_best_action(self, action_sequence: Tensor, returns: Tensor) -> Tensor: ...

    def update_sequence_generation(self, elite_actions: Tensor) -> None: ...
