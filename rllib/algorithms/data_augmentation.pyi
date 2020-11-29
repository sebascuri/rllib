from typing import Any, Optional, Union

from torch import Tensor
from torch.distributions import Distribution

from rllib.dataset.datatypes import Loss, Observation, Trajectory
from rllib.dataset.experience_replay import ExperienceReplay, StateExperienceReplay

from .dyna import Dyna

class DataAugmentation(Dyna):
    memory: Optional[ExperienceReplay] = ...
    initial_state_dataset: Optional[StateExperienceReplay] = ...
    initial_distribution: Optional[Distribution] = ...
    num_initial_state_samples: int = ...
    num_initial_distribution_samples: int = ...
    num_memory_samples: int = ...
    refresh_interval: int = ...
    sim_memory: ExperienceReplay
    count: int
    def __init__(
        self,
        memory: Optional[ExperienceReplay] = ...,
        initial_state_dataset: Optional[StateExperienceReplay] = ...,
        initial_distribution: Optional[Distribution] = ...,
        num_initial_state_samples: int = ...,
        num_initial_distribution_samples: int = ...,
        num_memory_samples: int = ...,
        refresh_interval: int = ...,
        only_sim: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def forward(
        self, observation: Union[Observation, Trajectory], **kwargs: Any
    ) -> Loss: ...
    def _sample_initial_states(self) -> Tensor: ...
