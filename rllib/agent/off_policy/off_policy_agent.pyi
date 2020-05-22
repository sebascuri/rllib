"""Off Policy Agent."""
from torch.optim.optimizer import Optimizer

from .abstract_agent import AbstractAgent
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


    def __init__(self, env_name: str,
                 optimizer: Optimizer,
                 memory: ExperienceReplay,
                 batch_size: int = 64,
                 target_update_frequency: int = 1,
                 num_iter: int = 1,
                 train_frequency: int = 1, num_rollouts: int = 0, gamma: float = 1.0,
                 exploration_steps: int = 0, exploration_episodes: int = 0,
                 comment: str = '') -> None: ...
