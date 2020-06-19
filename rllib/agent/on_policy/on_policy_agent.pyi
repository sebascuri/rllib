"""On Policy Agent."""
from typing import List

from torch.optim.optimizer import Optimizer

from rllib.agent.abstract_agent import AbstractAgent
from rllib.algorithms.abstract_algorithm import AbstractAlgorithm
from rllib.dataset.datatypes import Observation

class OnPolicyAgent(AbstractAgent):
    """Template for an on-policy algorithm."""

    algorithm: AbstractAlgorithm
    batch_size: int
    trajectories: List[List[Observation]]
    optimizer: Optimizer
    target_update_frequency: int
    num_iter: int
    def __init__(
        self,
        optimizer: Optimizer,
        batch_size: int = ...,
        target_update_frequency: int = ...,
        num_iter: int = ...,
        train_frequency: int = ...,
        num_rollouts: int = ...,
        gamma: float = ...,
        exploration_steps: int = ...,
        exploration_episodes: int = ...,
        tensorboard: bool = ...,
        comment: str = ...,
    ) -> None: ...
