from .abstract_solver import MPCSolver
from typing import Any

class GradientBasedSolver(MPCSolver):
    """Gradient based MPC solver."""
    lr: float
    def __init__(self, lr: float = ..., *args: Any, **kwargs: Any) -> None: ...
