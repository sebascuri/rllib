from torch import Tensor

from rllib.dataset.datatypes import TupleDistribution
from .abstract_policy import AbstractPolicy


class RandomPolicy(AbstractPolicy):
    def __init__(self, dim_state: int, dim_action: int,
                 num_states: int = None, num_actions: int = None) -> None: ...


    def forward(self, *args: Tensor, **kwargs) -> TupleDistribution: ...
