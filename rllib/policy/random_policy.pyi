from torch import Tensor

from rllib.dataset.datatypes import Action, TupleDistribution

from .abstract_policy import AbstractPolicy

class RandomPolicy(AbstractPolicy):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        num_states: int = -1,
        num_actions: int = -1,
        action_scale: Action = 1.0,
    ) -> None: ...
    def forward(self, *args: Tensor, **kwargs) -> TupleDistribution: ...
