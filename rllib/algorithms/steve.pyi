from typing import Optional

from torch import Tensor

from rllib.dataset.datatypes import Observation, Termination
from rllib.model import AbstractModel
from rllib.reward import AbstractReward
from rllib.value_function import AbstractValueFunction

from .abstract_algorithm import AbstractAlgorithm

class STEVE(AbstractAlgorithm):
    dynamical_model: AbstractModel
    reward_model: AbstractReward
    termination: Optional[Termination]
    value_target: Optional[AbstractValueFunction]
    num_steps: int
    num_samples: int
    def __init__(self, base_alg: AbstractAlgorithm) -> None: ...
    def get_value_target(self, observation: Observation) -> Tensor: ...

def steve_expand(
    base_algorithm: AbstractAlgorithm,
    dynamical_model: AbstractModel,
    reward_model: AbstractReward,
    num_steps: int = ...,
    num_samples: int = ...,
    termination: Optional[Termination] = ...,
) -> STEVE: ...
