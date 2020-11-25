from typing import Any, Optional

from torch.distributions import Distribution

from rllib.agent import AbstractAgent
from rllib.algorithms.simulation_algorithm import SimulationAlgorithm
from rllib.dataset.experience_replay import ExperienceReplay

from .model_based_agent import ModelBasedAgent

class DataAugmentationAgent(ModelBasedAgent):
    simulation_algorithm: SimulationAlgorithm
    sim_memory: ExperienceReplay
    refresh_interval: int
    only_sim: bool
    def __init__(
        self,
        base_agent: AbstractAgent,
        num_steps: int = ...,
        num_samples: int = ...,
        num_initial_distribution_samples: int = ...,
        num_memory_samples: int = ...,
        num_initial_state_samples: int = ...,
        refresh_interval: int = ...,
        initial_distribution: Optional[Distribution] = ...,
        only_sim: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def simulate(self) -> None: ...
