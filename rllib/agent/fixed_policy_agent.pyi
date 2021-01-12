from typing import Any

from rllib.policy import AbstractPolicy

from .abstract_agent import AbstractAgent

class FixedPolicyAgent(AbstractAgent):
    def __init__(self, policy: AbstractPolicy, *args: Any, **kwargs: Any) -> None: ...
