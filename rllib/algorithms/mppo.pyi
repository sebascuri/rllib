"""Maximum a Posterior Policy Optimization algorithm stub."""

from typing import List, Tuple, Union
from typing import NamedTuple, Callable

import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution
from torch.optim.optimizer import Optimizer

from rllib.model import AbstractModel
from rllib.policy import AbstractPolicy
from rllib.reward import AbstractReward
from rllib.value_function import AbstractValueFunction, AbstractQFunction
from rllib.util.parameter_decay import ParameterDecay


class MPOLosses(NamedTuple):
    primal_loss: Tensor
    dual_loss: Tensor


class MPOReturn(NamedTuple):
    loss: Tensor
    value_loss: Tensor
    primal_loss: Tensor
    dual_loss: Tensor
    kl_div: Tensor


class MPPOLoss(nn.Module):
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


class MPPO(nn.Module):
    old_policy: AbstractPolicy
    policy: AbstractPolicy
    q_function: AbstractQFunction
    gamma: float
    num_action_samples: int

    mppo_loss: MPPOLoss
    value_loss: nn.modules.loss._Loss

    def __init__(self, policy: AbstractPolicy, q_function: AbstractQFunction,
                 num_action_samples: int,
                 epsilon: Union[ParameterDecay, float] = None,
                 epsilon_mean: Union[ParameterDecay, float] = None,
                 epsilon_var: Union[ParameterDecay, float] = None,
                 eta: Union[ParameterDecay, float] = None,
                 eta_mean: Union[ParameterDecay, float] = None,
                 eta_var: Union[ParameterDecay, float] = None,
                 gamma: float = 0.99
                 ) -> None: ...

    def reset(self) -> None: ...

    def forward(self, *args: Tensor, **kwargs) -> MPOReturn: ...


class MBMPPO(nn.Module):
    dynamical_model: AbstractModel
    reward_model: AbstractReward
    policy: AbstractPolicy
    value_function: AbstractValueFunction

    gamma: float

    mppo_loss: MPPOLoss
    value_loss: nn.modules.loss._Loss
    num_action_samples: int
    entropy_reg: float
    termination: Callable[[Tensor, Tensor], Tensor]

    def __init__(self, dynamical_model: AbstractModel, reward_model: AbstractReward,
                 policy: AbstractPolicy, value_function: AbstractValueFunction,
                 epsilon: Union[ParameterDecay, float] = None,
                 epsilon_mean: Union[ParameterDecay, float] = None,
                 epsilon_var: Union[ParameterDecay, float] = None,
                 eta: Union[ParameterDecay, float] = None,
                 eta_mean: Union[ParameterDecay, float] = None,
                 eta_var: Union[ParameterDecay, float] = None,
                 gamma: float = 0.99,
                 num_action_samples: int = 15,
                 entropy_reg: float = 0.,
                 termination: Callable[[Tensor, Tensor], Tensor] = None) -> None: ...

    def reset(self) -> None: ...

    def forward(self, *args: Tensor, **kwargs) -> MPOReturn: ...


def train_mppo(mppo: MBMPPO, initial_distribution: Distribution, optimizer: Optimizer,
               num_iter: int, num_trajectories: int, num_simulation_steps: int,
               num_gradient_steps: int,
               batch_size: int, num_subsample: int
               ) -> Tuple[List, List, List, List, List]: ...
