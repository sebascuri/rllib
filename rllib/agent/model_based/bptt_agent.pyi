from typing import Any, Optional, Type, Union

from torch.nn.modules.loss import _Loss

from rllib.algorithms.bptt import BPTT
from rllib.model import AbstractModel
from rllib.policy import AbstractPolicy
from rllib.util.parameter_decay import ParameterDecay
from rllib.value_function import AbstractQFunction

from .model_based_agent import ModelBasedAgent

class BPTTAgent(ModelBasedAgent):
    algorithm: BPTT
    def __init__(
        self,
        policy: AbstractPolicy,
        critic: AbstractQFunction,
        dynamical_model: AbstractModel,
        reward_model: AbstractModel,
        criterion: Type[_Loss],
        termination_model: Optional[AbstractModel] = ...,
        algorithm: Type[BPTT] = ...,
        epsilon_mean: Union[ParameterDecay, float] = ...,
        epsilon_var: Optional[Union[ParameterDecay, float]] = ...,
        regularization: bool = ...,
        num_steps: int = ...,
        num_samples: int = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
