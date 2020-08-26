from typing import Any, List

from rllib.agent.abstract_agent import AbstractAgent
from rllib.dataset.datatypes import Observation

class OnPolicyAgent(AbstractAgent):
    """Template for an on-policy algorithm."""

    trajectories: List[List[Observation]]
    def __init__(self, num_rollouts: int = ..., *args: Any, **kwargs: Any,) -> None: ...
