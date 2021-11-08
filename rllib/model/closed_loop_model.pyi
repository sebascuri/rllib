"""Closed Loop Model file."""

from .abstract_model import AbstractModel
from rllib.policy.abstract_policy import AbstractPolicy
from typing import Any

class ClosedLoopModel(AbstractModel):
    base_model: AbstractModel
    policy: AbstractPolicy
    def __init__(
        self,
        base_model: AbstractModel,
        policy: AbstractPolicy,
        *args: Any,
        **kwargs: Any
    ) -> None: ...
