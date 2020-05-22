"""Maximum a Posterior Policy Optimization algorithm stub."""

from typing import List, Tuple, Union, Optional, NamedTuple

import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution
from torch.optim.optimizer import Optimizer
from torch.nn.modules.loss import _Loss

from .abstract_algorithm import AbstractAlgorithm, MPOLoss
from rllib.dataset.datatypes import Termination
from rllib.model import AbstractModel
from rllib.policy import AbstractPolicy
from rllib.reward import AbstractReward
from rllib.value_function import AbstractValueFunction, AbstractQFunction
from rllib.util.parameter_decay import ParameterDecay


class MPOLosses(NamedTuple):
    primal_loss: Tensor
    dual_loss: Tensor


class MPPOWorker(nn.Module):
    eta: ParameterDecay
    eta_mean: ParameterDecay
    eta_var: ParameterDecay

    epsilon: Tensor
    epsilon_mean: Tensor
    epsilon_var: Tensor

    def __init__(self, epsilon: Union[ParameterDecay, float] = None,
                 epsilon_mean: Union[ParameterDecay, float] = None,
                 epsilon_var: Union[ParameterDecay, float] = None,
                 eta: Union[ParameterDecay, float] = None,
                 eta_mean: Union[ParameterDecay, float] = None,
                 eta_var: Union[ParameterDecay, float] = None
                 ) -> None: ...

    def forward(self, *args: Tensor, **kwargs: Tensor) -> MPOLosses: ...


class MPPO(AbstractAlgorithm):
    old_policy: AbstractPolicy
    policy: AbstractPolicy
    q_function: AbstractQFunction
    q_target: AbstractQFunction

    gamma: float
    num_action_samples: int

    mppo_loss: MPPOWorker
    value_loss: _Loss
    entropy_reg: float

    def __init__(self, policy: AbstractPolicy, q_function: AbstractQFunction,
                 num_action_samples: int,
                 criterion: _Loss,
                 entropy_reg: float = 0.,
                 epsilon: Union[ParameterDecay, float] = None,
                 epsilon_mean: Union[ParameterDecay, float] = None,
                 epsilon_var: Union[ParameterDecay, float] = None,
                 eta: Union[ParameterDecay, float] = None,
                 eta_mean: Union[ParameterDecay, float] = None,
                 eta_var: Union[ParameterDecay, float] = None,
                 gamma: float = 0.99
                 ) -> None: ...

    def reset(self) -> None: ...

    def forward(self, *args: Tensor, **kwargs) -> MPOLoss: ...


class MBMPPO(AbstractAlgorithm):
    dynamical_model: AbstractModel
    reward_model: AbstractReward
    policy: AbstractPolicy
    value_function: AbstractValueFunction
    value_function_target: AbstractValueFunction

    gamma: float

    mppo_loss: MPPOWorker
    value_loss: _Loss
    num_action_samples: int
    entropy_reg: float
    termination: Optional[Termination]

    def __init__(self, dynamical_model: AbstractModel, reward_model: AbstractReward,
                 policy: AbstractPolicy, value_function: AbstractValueFunction,
                 criterion: _Loss,
                 epsilon: Union[ParameterDecay, float] = None,
                 epsilon_mean: Union[ParameterDecay, float] = None,
                 epsilon_var: Union[ParameterDecay, float] = None,
                 eta: Union[ParameterDecay, float] = None,
                 eta_mean: Union[ParameterDecay, float] = None,
                 eta_var: Union[ParameterDecay, float] = None,
                 gamma: float = 0.99,
                 num_action_samples: int = 15,
                 entropy_reg: float = 0.,
                 termination: Termination = None) -> None: ...

    def reset(self) -> None: ...

    def forward(self, *args: Tensor, **kwargs) -> MPOLoss: ...


def train_mppo(mppo: MBMPPO, initial_distribution: Distribution, optimizer: Optimizer,
               num_iter: int, num_trajectories: int, num_simulation_steps: int,
               num_gradient_steps: int,
               batch_size: int, num_subsample: int
               ) -> Tuple[List, List, List, List, List]: ...
