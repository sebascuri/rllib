from typing import Any, Tuple

from torch import Tensor

from rllib.dataset.datatypes import TupleDistribution

from .abstract_policy import AbstractPolicy

class DerivedPolicy(AbstractPolicy):

    base_policy: AbstractPolicy
    def __init__(
        self, base_policy: AbstractPolicy, dim_action: Tuple, *args: Any, **kwargs: Any
    ) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> TupleDistribution: ...
