"""Closed Loop Model file."""

from typing import Any

from rllib.policy.abstract_policy import AbstractPolicy

from .abstract_model import AbstractModel

class ClosedLoopModel(AbstractModel):
    base_model: AbstractModel
    policy: AbstractPolicy
    def __init__(
        self,
        base_model: AbstractModel,
        policy: AbstractPolicy,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
