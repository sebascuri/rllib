from typing import Any, Optional

from torch import Tensor

from .abstract_solver import MPCSolver

class CEMShooting(MPCSolver):
    num_elites: int
    alpha: float
    def __init__(
        self,
        alpha: float = ...,
        num_elites: Optional[int] = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def get_candidate_action_sequence(self) -> Tensor: ...
    def get_best_action(self, action_sequence: Tensor, returns: Tensor) -> Tensor: ...
    def update_sequence_generation(self, elite_actions: Tensor) -> None: ...
