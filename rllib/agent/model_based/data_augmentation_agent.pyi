from typing import Any, Optional

from torch.distributions import Distribution

from rllib.algorithms.abstract_algorithm import AbstractAlgorithm

from .model_based_agent import ModelBasedAgent

class DataAugmentationAgent(ModelBasedAgent):
    def __init__(
        self,
        base_algorithm: AbstractAlgorithm,
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
