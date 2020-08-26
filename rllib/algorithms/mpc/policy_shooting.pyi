from typing import Any

from rllib.policy import AbstractPolicy

from .random_shooting import RandomShooting

class PolicyShooting(RandomShooting):
    def __init__(self, policy: AbstractPolicy, *args: Any, **kwargs: Any) -> None: ...
