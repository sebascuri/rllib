"""Python Script Template."""

from typing import List, Tuple
from typing import NamedTuple, Callable

import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution
from torch.optim.optimizer import Optimizer

from rllib.model import AbstractModel
from rllib.policy import AbstractPolicy
from rllib.reward import AbstractReward
from rllib.value_function import AbstractValueFunction


class MPOLosses(NamedTuple):
    primal_loss: Tensor
    dual_loss: Tensor

class MPOReturn(NamedTuple):
    loss: Tensor
    value_loss: Tensor
    primal_loss: Tensor
    dual_loss: Tensor
    kl_div: Tensor


class MPPO(nn.Module):
    eta: nn.Parameter
    eta_mean: nn.Parameter
    eta_var: nn.Parameter

    epsilon: Tensor
    epsilon_mean: Tensor
    epsilon_var: Tensor

    def __init__(self, epsilon: float, epsilon_mean: float, epsilon_var: float
                 ) -> None: ...

    def forward(self, *args: Tensor, **kwargs: Tensor) -> MPOLosses: ...


class MBMPPO(nn.Module):
    dynamical_model: AbstractModel
    reward_model: AbstractReward
    policy: AbstractPolicy
    value_function: AbstractValueFunction

    gamma: float

    mppo: MPPO
    value_loss: nn.modules.loss._Loss
    num_action_samples: int
    termination: Callable[[Tensor, Tensor], Tensor]

    def __init__(self, dynamical_model: AbstractModel, reward_model: AbstractReward,
                 policy: AbstractPolicy, value_function: AbstractValueFunction,
                 epsilon: float, epsilon_mean: float, epsilon_var: float, gamma: float,
                 num_action_samples: int = 15,
                 termination: Callable[[Tensor, Tensor], Tensor] = None) -> None: ...

    def reset(self) -> None: ...

    def forward(self, *args: Tensor, **kwargs) -> MPOReturn: ...


def train_mppo(mppo: MBMPPO, initial_distribution: Distribution, optimizer: Optimizer,
               num_iter: int, num_trajectories: int, num_simulation_steps: int,
               refresh_interval: int,
               batch_size: int, num_subsample: int) -> Tuple[List, List, List, List, List]: ...