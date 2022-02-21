from typing import Any, Optional

import torch
from torch.distributions import Distribution

from rllib.agent.abstract_agent import AbstractAgent
from rllib.algorithms.abstract_algorithm import AbstractAlgorithm
from rllib.algorithms.model_learning_algorithm import ModelLearningAlgorithm
from rllib.dataset.experience_replay import ExperienceReplay, StateExperienceReplay
from rllib.model import AbstractModel
from rllib.policy import AbstractPolicy

class ModelBasedAgent(AbstractAgent):
    model_learning_algorithm: ModelLearningAlgorithm
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
    simulation_frequency: int
    simulation_max_steps: int = ...
    num_memory_samples: int = ...
    num_initial_state_samples: int = ...
    num_initial_distribution_samples: int = ...
    initial_distribution: Optional[Distribution]
    augment_dataset_with_sim: bool
    pre_train_iterations: int
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
        memory: Optional[ExperienceReplay] = ...,
        policy: Optional[AbstractPolicy] = ...,
        thompson_sampling: bool = ...,
        simulation_frequency: int = ...,
        simulation_max_steps: int = ...,
        num_memory_samples: int = ...,
        num_initial_state_samples: int = ...,
        num_initial_distribution_samples: int = ...,
        initial_distribution: Optional[Distribution] = ...,
        augment_dataset_with_sim: bool = ...,
        pre_train_iterations: int = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    @property
    def learn_model_at_observe(self) -> bool: ...
    @property
    def learn_model_at_end_episode(self) -> bool: ...
    @property
    def pretrain_model(self) -> bool: ...
    def simulate_policy_on_model(self) -> None: ...
    def _sample_initial_states(self) -> torch.Tensor: ...
    @property
    def simulate(self) -> bool: ...
