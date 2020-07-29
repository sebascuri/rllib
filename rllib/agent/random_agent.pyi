from typing import List, Tuple

from rllib.dataset import TrajectoryDataset
from rllib.dataset.datatypes import Observation
from rllib.policy import RandomPolicy

from .abstract_agent import AbstractAgent

class RandomAgent(AbstractAgent):
    """Agent that interacts randomly in an environment."""

    policy: RandomPolicy
    trajectory: List[Observation]
    dataset: TrajectoryDataset
    def __init__(
        self,
        dim_state: Tuple,
        dim_action: Tuple,
        num_states: int = ...,
        num_actions: int = ...,
        train_frequency: int = ...,
        num_rollouts: int = ...,
        gamma: float = ...,
        exploration_steps: int = ...,
        exploration_episodes: int = ...,
        tensorboard: bool = ...,
        comment: str = ...,
    ) -> None: ...
