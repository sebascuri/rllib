from typing import Any, Optional

from torch import Tensor

from rllib.algorithms.mpc import MPCSolver

from .abstract_policy import AbstractPolicy

class MPCPolicy(AbstractPolicy):

    solver: MPCSolver
    _steps: int
    solver_frequency: int
    action_sequence: Optional[Tensor]
    def __init__(
        self,
        mpc_solver: MPCSolver,
        solver_frequency: int = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
