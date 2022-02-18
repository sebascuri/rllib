from abc import ABCMeta
from typing import Any, Optional

from rllib.model import AbstractModel

from .abstract_algorithm import AbstractAlgorithm
from .simulation_algorithm import SimulationAlgorithm

class AbstractMBAlgorithm(AbstractAlgorithm, metaclass=ABCMeta):
    dynamical_model: AbstractModel
    reward_model: AbstractModel
    termination_model: Optional[AbstractModel]
    num_model_steps: int
    num_particles: int
    log_simulation: bool
    simulation_algorithm: SimulationAlgorithm
    # _info: dict
    def __init__(
        self,
        dynamical_model: AbstractModel,
        reward_model: AbstractModel,
        num_model_steps: int = ...,
        num_particles: int = ...,
        termination_model: Optional[AbstractModel] = ...,
        log_simulation: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
