from typing import Any, Optional

from torch import Tensor
from torch.distributions import Distribution

from rllib.dataset.datatypes import Observation
from rllib.dataset.experience_replay import ExperienceReplay, StateExperienceReplay
from rllib.policy import AbstractPolicy
from rllib.util.logger import Logger

from .abstract_mb_algorithm import AbstractMBAlgorithm

class SimulationAlgorithm(AbstractMBAlgorithm):
    initial_distribution: Optional[Distribution]
    num_subsample: int
    num_initial_state_samples: int
    num_initial_distribution_samples: int
    num_memory_samples: int
    refresh_interval: int
    _idx: int
    dataset: StateExperienceReplay
    def __init__(
        self,
        initial_distribution: Optional[Distribution] = ...,
        max_memory: int = ...,
        num_subsample: int = ...,
        num_initial_state_samples: int = ...,
        num_initial_distribution_samples: int = ...,
        num_memory_samples: int = ...,
        refresh_interval: int = ...,
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
    ) -> Observation: ...
    def _log_trajectory(
        self, stacked_trajectory: Observation, logger: Optional[Logger] = ...
    ) -> None: ...
