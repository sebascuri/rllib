from typing import Optional

from rllib.agent.abstract_agent import AbstractAgent
from rllib.algorithms.abstract_algorithm import AbstractAlgorithm
from rllib.algorithms.model_learning_algorithm import ModelLearningAlgorithm
from rllib.algorithms.mpc.abstract_solver import MPCSolver
from rllib.algorithms.simulation_algorithm import SimulationAlgorithm
from rllib.dataset.datatypes import Trajectory
from rllib.model import AbstractModel
from rllib.policy import AbstractPolicy
from rllib.policy.derived_policy import DerivedPolicy
from rllib.reward import AbstractReward

class ModelBasedAgent(AbstractAgent):
    policy: DerivedPolicy
    algorithm: AbstractAlgorithm
    model_learning_algorithm: ModelLearningAlgorithm
    planning_algorithm: Optional[MPCSolver]
    simulation_algorithm: Optional[SimulationAlgorithm]
    dynamical_model: AbstractModel
    reward_model: AbstractReward
    num_simulation_iterations: int
    learn_from_real: bool
    learn_from_sim: bool
    thompson_sampling: bool
    def __init__(
        self,
        policy: AbstractPolicy,
        policy_learning_algorithm: Optional[AbstractAlgorithm] = ...,
        model_learning_algorithm: Optional[ModelLearningAlgorithm] = ...,
        planning_algorithm: Optional[MPCSolver] = ...,
        simulation_algorithm: Optional[SimulationAlgorithm] = ...,
        num_simulation_iterations: int = ...,
        learn_from_real: bool = ...,
        thompson_sampling: bool = ...,
    ) -> None: ...
    def learn(self) -> None: ...
    def log_trajectory(self, trajectory: Trajectory) -> None: ...
    def simulate_and_learn_policy(self): ...
    def learn_policy_from_sim_data(self): ...
    def learn_policy_from_real_data(self): ...
