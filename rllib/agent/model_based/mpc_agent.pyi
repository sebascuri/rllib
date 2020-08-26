from typing import Any

from rllib.algorithms.mpc.abstract_solver import MPCSolver

from .model_based_agent import ModelBasedAgent

class MPCAgent(ModelBasedAgent):
    """Implementation of an agent that runs an MPC policy."""

    def __init__(self, mpc_solver: MPCSolver, *args: Any, **kwargs: Any) -> None: ...
