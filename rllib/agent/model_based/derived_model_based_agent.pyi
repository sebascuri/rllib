from typing import Any, Callable, Optional

from rllib.algorithms.abstract_algorithm import AbstractAlgorithm
from rllib.dataset.datatypes import Termination
from rllib.model import AbstractModel
from rllib.reward import AbstractReward

from .model_based_agent import ModelBasedAgent

class DerivedMBAgent(ModelBasedAgent):
    def __init__(
        self,
        base_algorithm: AbstractAlgorithm,
        derived_algorithm_: Callable,
        dynamical_model: AbstractModel,
        reward_model: AbstractReward,
        num_samples: int = ...,
        num_steps: int = ...,
        termination: Optional[Termination] = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
