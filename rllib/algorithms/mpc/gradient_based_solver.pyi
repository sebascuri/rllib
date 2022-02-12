from typing import Any

from torch import Tensor

from .abstract_solver import MPCSolver

class GradientBasedSolver(MPCSolver):
    """Gradient based MPC solver."""

    lr: float
    def __init__(self, lr: float = ..., *args: Any, **kwargs: Any) -> None: ...
    def get_candidate_action_sequence(self) -> Tensor: ...
    def get_best_action(self, action_sequence: Tensor, returns: Tensor) -> Tensor: ...
    def update_sequence_generation(self, elite_actions: Tensor) -> None: ...
