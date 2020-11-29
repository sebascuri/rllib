from typing import Any, Optional, Union

from torch import Tensor
from torch.distributions import Distribution

from rllib.dataset.datatypes import Observation, Trajectory
from rllib.dataset.experience_replay import ExperienceReplay, StateExperienceReplay
from rllib.policy import AbstractPolicy
from rllib.util.logger import Logger

from .abstract_mb_algorithm import AbstractMBAlgorithm

class SimulationAlgorithm(AbstractMBAlgorithm):
    initial_distribution: Optional[Distribution]
    num_initial_state_samples: int
    num_initial_distribution_samples: int
    num_memory_samples: int
    def __init__(
        self,
        initial_distribution: Optional[Distribution] = ...,
        num_initial_state_samples: int = ...,
        num_initial_distribution_samples: int = ...,
        num_memory_samples: int = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def get_initial_states(
        self,
        initial_states_dataset: StateExperienceReplay,
        real_dataset: ExperienceReplay,
    ) -> Tensor: ...
    def simulate(
        self,
        state: Tensor,
        policy: AbstractPolicy,
        initial_action: Optional[Tensor] = ...,
        logger: Optional[Logger] = ...,
        stack_obs: bool = ...,
    ) -> Union[Observation, Trajectory]: ...
