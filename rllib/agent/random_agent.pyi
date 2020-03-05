from .abstract_agent import AbstractAgent
from rllib.dataset import TrajectoryDataset
from rllib.dataset.datatypes import Observation
from rllib.policy import RandomPolicy
from typing import List

class RandomAgent(AbstractAgent):
    """Agent that interacts randomly in an environment."""
    policy: RandomPolicy
    trajectory: List[Observation]
    dataset: TrajectoryDataset

    def __init__(self, dim_state: int, dim_action: int,
                 num_states: int = None, num_actions: int = None,
                 gamma: float = None,
                 exploration_steps: int = 0, exploration_episodes: int = 0) -> None: ...
