from typing import Any

from rllib.agent.abstract_agent import AbstractAgent
from rllib.algorithms.abstract_algorithm import AbstractAlgorithm
from rllib.dataset.experience_replay import ExperienceReplay

class OffPolicyAgent(AbstractAgent):
    """Template for an on-policy algorithm."""

    algorithm: AbstractAlgorithm
    memory: ExperienceReplay
    def __init__(
        self,
        memory: ExperienceReplay,
        num_iter: int = ...,
        batch_size: int = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
