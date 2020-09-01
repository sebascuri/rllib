from typing import Any, Callable, Optional

from rllib.algorithms.abstract_algorithm import AbstractAlgorithm
from rllib.model import AbstractModel

from .model_based_agent import ModelBasedAgent

class DerivedMBAgent(ModelBasedAgent):
    def __init__(
        self,
        base_algorithm: AbstractAlgorithm,
        derived_algorithm_: Callable,
        dynamical_model: AbstractModel,
        reward_model: AbstractModel,
        num_samples: int = ...,
        num_steps: int = ...,
        termination_model: Optional[AbstractModel] = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
