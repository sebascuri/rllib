from typing import Any, Optional, Tuple

from torch import Tensor

from rllib.dataset.datatypes import Action, TupleDistribution

from .abstract_policy import AbstractPolicy

class RandomPolicy(AbstractPolicy):
    def __init__(
        self,
        dim_state: Tuple,
        dim_action: Tuple,
        num_states: int = ...,
        num_actions: int = ...,
        action_scale: Action = ...,
        goal: Optional[Tensor] = ...,
    ) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> TupleDistribution: ...
