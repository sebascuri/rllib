from typing import Any, Optional, Union

from torch import Tensor

from rllib.dataset.datatypes import Observation, Trajectory
from rllib.model import AbstractModel
from rllib.policy import AbstractPolicy
from rllib.util.logger import Logger

class AbstractMBAlgorithm(object):
    dynamical_model: AbstractModel
    reward_model: AbstractModel
    termination_model: Optional[AbstractModel]
    num_steps: int
    # num_samples: int
    log_simulation: bool
    # _info: dict
    def __init__(
        self,
        dynamical_model: AbstractModel,
        reward_model: AbstractModel,
        num_steps: int = ...,
        num_samples: int = ...,
        termination_model: Optional[AbstractModel] = ...,
        log_simulation: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def simulate(
        self,
        state: Tensor,
        policy: AbstractPolicy,
        initial_action: Optional[Tensor] = ...,
        logger: Optional[Logger] = ...,
        stack_obs: bool = ...,
    ) -> Union[Observation, Trajectory]: ...
    def _log_trajectory(self, trajectory: Trajectory) -> None: ...
    def _log_observation(self, observation: Observation) -> None: ...
