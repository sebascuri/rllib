"""Model-Based MPO Agent Implementation."""
from typing import Any, Optional, Type, Union

from torch.nn.modules.loss import _Loss

from rllib.policy import AbstractPolicy
from rllib.util.parameter_decay import ParameterDecay
from rllib.value_function import AbstractQFunction

import torch.nn.modules.loss as loss
from rllib.algorithms.mb_mpo import MBMPO

from .model_based_agent import ModelBasedAgent

class MBMPOAgent(ModelBasedAgent):
    """Implementation of an agent that runs MB-MPO."""

    algorithm: MBMPO
    def __init__(
        self,
        policy: AbstractPolicy,
        critic: AbstractQFunction,
        criterion: Type[_Loss] = ...,
        num_action_samples: int = ...,
        epsilon: Union[ParameterDecay, float] = ...,
        epsilon_mean: Union[ParameterDecay, float] = ...,
        epsilon_var: Optional[Union[ParameterDecay, float]] = ...,
        kl_regularization: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
