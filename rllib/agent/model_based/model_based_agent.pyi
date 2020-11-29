from typing import Any, Optional

from rllib.agent.abstract_agent import AbstractAgent
from rllib.algorithms.abstract_algorithm import AbstractAlgorithm
from rllib.algorithms.model_learning_algorithm import ModelLearningAlgorithm
from rllib.algorithms.mpc.abstract_solver import MPCSolver
from rllib.dataset.experience_replay import ExperienceReplay, StateExperienceReplay
from rllib.model import AbstractModel

class ModelBasedAgent(AbstractAgent):
    model_learning_algorithm: ModelLearningAlgorithm
    planning_algorithm: Optional[MPCSolver]
    dynamical_model: AbstractModel
    reward_model: AbstractModel
    termination_model: Optional[AbstractModel]
    thompson_sampling: bool
    memory: ExperienceReplay
    initial_states_dataset: StateExperienceReplay
    model_learn_train_frequency: int
    model_learn_num_rollouts: int
    model_learn_exploration_episodes: int
    model_learn_exploration_steps: int
    def __init__(
        self,
        dynamical_model: AbstractModel,
        reward_model: AbstractModel,
        termination_model: Optional[AbstractModel] = ...,
        train_frequency: int = ...,
        num_rollouts: int = ...,
        model_learn_train_frequency: int = ...,
        model_learn_num_rollouts: int = ...,
        model_learn_exploration_episodes: Optional[int] = ...,
        model_learn_exploration_steps: Optional[int] = ...,
        policy_learning_algorithm: Optional[AbstractAlgorithm] = ...,
        model_learning_algorithm: Optional[ModelLearningAlgorithm] = ...,
        planning_algorithm: Optional[MPCSolver] = ...,
        memory: Optional[ExperienceReplay] = ...,
        thompson_sampling: bool = ...,
        training_verbose: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    @property
    def learn_model_at_observe(self) -> bool: ...
    @property
    def learn_model_at_end_episode(self) -> bool: ...
