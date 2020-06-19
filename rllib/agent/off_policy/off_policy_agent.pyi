"""Off Policy Agent."""
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
        optimizer: Optimizer,
        memory: ExperienceReplay,
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
