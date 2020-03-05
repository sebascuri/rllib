"""Python Script Template."""

import torch.nn as nn
from torch import Tensor
from typing import NamedTuple

from rllib.model import AbstractModel
from rllib.policy import AbstractPolicy
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
    def __init__(self, model: AbstractModel, reward_function,
                 policy: AbstractPolicy, value_function: AbstractValueFunction,
                 epsilon: float, epsilon_mean: float, epsilon_var: float, gamma: float,
                 num_action_samples: int = 15) -> None: ...

    def reset(self) -> None: ...

    def forward(self, *args: Tensor, **kwargs) -> MPOReturn: ...