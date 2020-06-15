"""Policy that Implements MPC."""

from rllib.algorithms.mpc import MPCSolver

from .abstract_policy import AbstractPolicy

class MPCPolicy(AbstractPolicy):
    """MPC Policy."""

    solver: MPCSolver
    def __init__(self, mpc_solver: MPCSolver) -> None: ...
