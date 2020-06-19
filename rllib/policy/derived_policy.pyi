from torch import Tensor

from rllib.dataset.datatypes import TupleDistribution

from .abstract_policy import AbstractPolicy

class DerivedPolicy(AbstractPolicy):

    base_policy: AbstractPolicy
    def __init__(self, base_policy: AbstractPolicy, dim_action: int) -> None: ...
    def forward(self, *args: Tensor, **kwargs) -> TupleDistribution: ...
