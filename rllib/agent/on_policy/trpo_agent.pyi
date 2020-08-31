from typing import Any, Optional, Union

import torch.nn.modules.loss as loss

from rllib.algorithms.trpo import TRPO
from rllib.policy import AbstractPolicy
from rllib.util.parameter_decay import ParameterDecay
from rllib.value_function import AbstractValueFunction

from .actor_critic_agent import ActorCriticAgent

class TRPOAgent(ActorCriticAgent):

    algorithm: TRPO  # type: ignore
    def __init__(
        self,
        policy: AbstractPolicy,
        critic: AbstractValueFunction,
        regularization: bool = ...,
        epsilon_mean: Union[float, ParameterDecay] = ...,
        epsilon_var: Optional[Union[float, ParameterDecay]] = ...,
        lambda_: float = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
