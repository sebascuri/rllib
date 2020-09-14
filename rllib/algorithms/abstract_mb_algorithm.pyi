from typing import Any, Optional, Tuple

from torch import Tensor

from rllib.dataset.datatypes import Observation
from rllib.model import AbstractModel
from rllib.policy import AbstractPolicy
from rllib.util.logger import Logger

class AbstractMBAlgorithm(object):
    dynamical_model: AbstractModel
    reward_model: AbstractModel
    termination_model: Optional[AbstractModel]
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
    def simulate(
        self,
        state: Tensor,
        policy: AbstractPolicy,
        initial_action: Optional[Tensor] = ...,
        logger: Optional[Logger] = ...,
    ) -> Observation: ...
