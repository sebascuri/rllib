from typing import Any

from rllib.environment import AbstractEnvironment
from rllib.policy import RandomPolicy

from .abstract_agent import AbstractAgent

class RandomAgent(AbstractAgent):
    """Agent that interacts randomly in an environment."""

    policy: RandomPolicy
    def __init__(
        self, environment: AbstractEnvironment, *args: Any, **kwargs: Any,
    ) -> None: ...
