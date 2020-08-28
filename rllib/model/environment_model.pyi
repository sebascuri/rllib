"""Model implemented by querying an environment."""
from typing import Any

from torch import Tensor

from rllib.dataset.datatypes import TupleDistribution
from rllib.environment import AbstractEnvironment

from .abstract_model import AbstractModel

class EnvironmentModel(AbstractModel):
    environment: AbstractEnvironment
    def __init__(
        self, environment: AbstractEnvironment, *args: Any, **kwargs: Any
    ) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> TupleDistribution: ...
