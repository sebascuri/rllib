from typing import Any, Optional, Tuple

from torch import Tensor

from rllib.dataset.datatypes import Action, TupleDistribution

from .abstract_policy import AbstractPolicy

class RandomPolicy(AbstractPolicy):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> TupleDistribution: ...
