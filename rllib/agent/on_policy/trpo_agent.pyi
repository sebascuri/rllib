from typing import Any, Optional, Type, Union

import torch.nn.modules.loss as loss
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from rllib.algorithms.trpo import TRPO
from rllib.policy import AbstractPolicy
from rllib.util.parameter_decay import ParameterDecay
from rllib.value_function import AbstractValueFunction

from .on_policy_agent import OnPolicyAgent

class TRPOAgent(OnPolicyAgent):

    algorithm: TRPO
    def __init__(
        self,
        policy: AbstractPolicy,
        value_function: AbstractValueFunction,
        optimizer: Optimizer,
        criterion: Type[_Loss],
        regularization: bool = ...,
        epsilon_mean: Union[float, ParameterDecay] = ...,
        epsilon_var: Optional[Union[float, ParameterDecay]] = ...,
        lambda_: float = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
