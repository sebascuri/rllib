"""Policy that Implements MPC."""

from .abstract_policy import AbstractPolicy
from rllib.algorithms.mpc import MPCSolver


class MPCPolicy(AbstractPolicy):
    """MPC Policy."""
    solver: MPCSolver

    def __init__(self, mpc_solver: MPCSolver) -> None: ...
