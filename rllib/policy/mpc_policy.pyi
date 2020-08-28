from typing import Any

from rllib.algorithms.mpc import MPCSolver

from .abstract_policy import AbstractPolicy

class MPCPolicy(AbstractPolicy):

    solver: MPCSolver
    def __init__(self, mpc_solver: MPCSolver, *args: Any, **kwargs: Any) -> None: ...
