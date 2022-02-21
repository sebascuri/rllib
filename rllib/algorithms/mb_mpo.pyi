"""Model-Based implementation of Maximum a Posterior Policy Optimization algorithm."""
from typing import Any, Optional
from .mpo import MPO
from .simulation_algorithm import SimulationAlgorithm
from rllib.model.abstract_model import AbstractModel

class MBMPO(MPO):
    """Model-Based implementation of Maximum a Posteriori Policy Optimizaiton."""

    simulator: SimulationAlgorithm
    def __init__(
        self,
        dynamical_model: AbstractModel,
        reward_model: AbstractModel,
        termination_model: Optional[AbstractModel] = ...,
        num_particles: int = ...,
        num_model_steps: int = ...,
        *args: Any,
        **kwargs: Any
    ) -> None: ...
