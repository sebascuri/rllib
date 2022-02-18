from .abstract_model import AbstractModel
from typing import Any

from torch import Tensor

from rllib.dataset.datatypes import TupleDistribution

from .abstract_model import AbstractModel

class EmptyModel(AbstractModel):
    """Empty model."""

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> TupleDistribution: ...
