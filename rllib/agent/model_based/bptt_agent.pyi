from typing import Any, Optional, Type

from torch.nn.modules.loss import _Loss

from rllib.algorithms.bptt import BPTT
from rllib.dataset.datatypes import Termination
from rllib.model import AbstractModel
from rllib.policy import AbstractPolicy
from rllib.reward import AbstractReward
from rllib.value_function import AbstractQFunction

from .model_based_agent import ModelBasedAgent

class BPTTAgent(ModelBasedAgent):
    algorithm: BPTT
    def __init__(
        self,
        policy: AbstractPolicy,
        critic: AbstractQFunction,
        dynamical_model: AbstractModel,
        reward_model: AbstractReward,
        criterion: Type[_Loss],
        termination: Optional[Termination] = ...,
        algorithm: Type[BPTT] = ...,
        num_steps: int = ...,
        num_samples: int = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
