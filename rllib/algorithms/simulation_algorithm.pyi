from typing import Optional

from torch import Tensor

from rllib.dataset.datatypes import Trajectory
from rllib.dataset.experience_replay import ExperienceReplay
from rllib.model.abstract_model import AbstractModel
from rllib.policy import AbstractPolicy

class SimulationAlgorithm(object):
    dynamical_model: AbstractModel
    reward_model: AbstractModel
    termination_model: Optional[AbstractModel]
    num_particles: int
    num_model_steps: int
    def __init__(
        self,
        dynamical_model: AbstractModel,
        reward_model: AbstractModel,
        termination_model: Optional[AbstractModel] = ...,
        num_particles: int = ...,
        num_model_steps: int = ...,
    ) -> None: ...
    def simulate(
        self,
        state: Tensor,
        policy: AbstractPolicy,
        initial_action: Optional[Tensor] = ...,
        memory: Optional[ExperienceReplay] = ...,
        stack_obs: bool = ...,
    ) -> Trajectory: ...
