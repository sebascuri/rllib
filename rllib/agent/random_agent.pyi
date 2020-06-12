from typing import List

from rllib.dataset import TrajectoryDataset
from rllib.dataset.datatypes import Observation
from rllib.policy import RandomPolicy

from .abstract_agent import AbstractAgent


class RandomAgent(AbstractAgent):
    """Agent that interacts randomly in an environment."""
    policy: RandomPolicy
    trajectory: List[Observation]
    dataset: TrajectoryDataset

    def __init__(self, dim_state: int, dim_action: int,
                 num_states: int = -1, num_actions: int = -1,
                 train_frequency: int = 0, num_rollouts: int = 1, gamma: float = 1.0,
                 exploration_steps: int = 0, exploration_episodes: int = 0,
                 comment: str = '') -> None: ...
