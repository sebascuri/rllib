"""Closed Loop Model file."""

from typing import Any

from rllib.policy.abstract_policy import AbstractPolicy

from .transformed_model import TransformedModel

class ClosedLoopModel(TransformedModel):
    base_model: AbstractModel
    policy: AbstractPolicy
    def __init__(
        self,
        base_model: AbstractModel,
        policy: AbstractPolicy,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
