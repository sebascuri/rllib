"""Off Policy Agent."""

from .abstract_agent import AbstractAgent
from rllib.algorithms.abstract_algorithm import AbstractAlgorithm
from rllib.dataset.experience_replay import ExperienceReplay


class OffPolicyAgent(AbstractAgent):
    """Template for an on-policy algorithm."""

    algorithm: AbstractAlgorithm
    memory: ExperienceReplay
    batch_size: int
    train_frequency: int

    def __init__(self, env_name: str,
                 memory: ExperienceReplay,
                 batch_size: int = 64,
                 train_frequency: int = 1,
                 gamma: float = 1.0,
                 exploration_steps: int = 0,
                 exploration_episodes: int = 0, comment: str = '') -> None: ...
