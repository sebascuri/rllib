from typing import Any, List

from rllib.agent.abstract_agent import AbstractAgent
from rllib.dataset.datatypes import Observation

class OnPolicyAgent(AbstractAgent):
    """Template for an on-policy algorithm."""

    batch_size: int
    trajectories: List[List[Observation]]
    num_iter: int
    def __init__(
        self,
        batch_size: int = ...,
        num_iter: int = ...,
        num_rollouts: int = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
