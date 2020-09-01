from typing import Any, Optional

from torch import Tensor

from rllib.dataset.datatypes import Trajectory
from rllib.model import AbstractModel
from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractValueFunction

class AbstractMBAlgorithm(object):
    dynamical_model: AbstractModel
    reward_model: AbstractModel
    termination_model: Optional[AbstractModel]
    value_target: Optional[AbstractValueFunction]
    num_steps: int
    num_samples: int
    def __init__(
        self,
        dynamical_model: AbstractModel,
        reward_model: AbstractModel,
        num_steps: int = ...,
        num_samples: int = ...,
        termination_model: Optional[AbstractModel] = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def simulate(self, state: Tensor, policy: AbstractPolicy) -> Trajectory: ...
