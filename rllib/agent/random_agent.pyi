from typing import Any, List, Tuple

from rllib.dataset import TrajectoryDataset
from rllib.dataset.datatypes import Observation
from rllib.policy import RandomPolicy

from .abstract_agent import AbstractAgent

class RandomAgent(AbstractAgent):
    """Agent that interacts randomly in an environment."""

    policy: RandomPolicy
    dataset: TrajectoryDataset
    def __init__(
        self,
        dim_state: Tuple,
        dim_action: Tuple,
        num_states: int = ...,
        num_actions: int = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
