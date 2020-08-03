from typing import Any

from torch.optim.optimizer import Optimizer

from rllib.agent.abstract_agent import AbstractAgent
from rllib.algorithms.abstract_algorithm import AbstractAlgorithm
from rllib.dataset.experience_replay import ExperienceReplay

class OffPolicyAgent(AbstractAgent):
    """Template for an on-policy algorithm."""

    algorithm: AbstractAlgorithm
    optimizer: Optimizer
    memory: ExperienceReplay
    batch_size: int
    target_update_frequency: int
    num_iter: int
    def __init__(
        self,
        memory: ExperienceReplay,
        optimizer: Optimizer,
        num_iter: int = ...,
        batch_size: int = ...,
        target_update_frequency: int = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
