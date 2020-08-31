from typing import Any

from rllib.dataset import TrajectoryDataset
from rllib.environment import AbstractEnvironment
from rllib.policy import RandomPolicy

from .abstract_agent import AbstractAgent

class RandomAgent(AbstractAgent):
    """Agent that interacts randomly in an environment."""

    policy: RandomPolicy
    dataset: TrajectoryDataset
    def __init__(
        self, environment: AbstractEnvironment, *args: Any, **kwargs: Any,
    ) -> None: ...
