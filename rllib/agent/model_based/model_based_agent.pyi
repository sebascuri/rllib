from typing import Optional

from torch import Tensor
from torch.distributions import Distribution
from torch.optim.optimizer import Optimizer

from rllib.agent.abstract_agent import AbstractAgent
from rllib.algorithms.abstract_algorithm import AbstractAlgorithm
from rllib.dataset.datatypes import Observation, Termination
from rllib.dataset.experience_replay import (ExperienceReplay,
                                             StateExperienceReplay)
from rllib.model import AbstractModel
from rllib.model.derived_model import TransformedModel
from rllib.policy import AbstractPolicy
from rllib.policy.derived_policy import DerivedPolicy
from rllib.reward import AbstractReward
from rllib.value_function import AbstractValueFunction


class ModelBasedAgent(AbstractAgent):
    dynamical_model: TransformedModel
    reward_model: AbstractReward
    termination: Optional[Termination]
    value_function: AbstractValueFunction

    model_optimizer: Optimizer
    dataset: ExperienceReplay
    sim_dataset: StateExperienceReplay
    sim_trajectory: Observation

    model_learn_num_iter: int
    model_learn_batch_size: int
    plan_horizon: int
    plan_samples: int
    plan_elites: int

    algorithm: AbstractAlgorithm
    policy: DerivedPolicy
    plan_policy: AbstractPolicy
    policy_opt_num_iter: int
    policy_opt_batch_size: int
    policy_opt_gradient_steps: int
    policy_opt_target_update_frequency: int
    optimizer: Optional[Optimizer]

    sim_num_steps: int
    sim_initial_states_num_trajectories: int
    sim_initial_dist_num_trajectories: int
    sim_memory_num_trajectories: int
    sim_refresh_interval: int
    sim_num_subsample: int
    initial_distribution: Distribution
    initial_states: StateExperienceReplay
    new_episode: bool
    thompson_sampling: bool

    def __init__(self,
                 dynamical_model: AbstractModel,
                 reward_model: AbstractReward,
                 policy: AbstractPolicy,
                 model_optimizer: Optimizer = None,
                 value_function: AbstractValueFunction = None,
                 termination: Termination = None,
                 plan_horizon: int = 1,
                 plan_samples: int = 1,
                 plan_elites: int = 1,
                 model_learn_num_iter: int = 0,
                 model_learn_batch_size: int = 64,
                 bootstrap: bool = True,
                 max_memory: int = 10000,
                 policy_opt_num_iter: int = 0,
                 policy_opt_batch_size: int = None,
                 policy_opt_gradient_steps: int = 0,
                 policy_opt_target_update_frequency: int = 1,
                 optimizer: Optimizer = None,
                 sim_num_steps: int = 20,
                 sim_initial_states_num_trajectories: int = 8,
                 sim_initial_dist_num_trajectories: int = 0,
                 sim_memory_num_trajectories: int = 0,
                 sim_refresh_interval: int = 1,
                 sim_num_subsample: int = 1,
                 sim_max_memory: int = 10000,
                 initial_distribution: Distribution = None,
                 thompson_sampling: bool = False,
                 train_frequency: int = 0, num_rollouts: int = 1, gamma: float = 1.0,
                 exploration_steps: int = 0, exploration_episodes: int = 0,
                 tensorboard: bool = False,
                 comment: str = '') -> None: ...

    def _plan(self, state: Tensor) -> None: ...

    def _train(self) -> None: ...

    def _train_model(self) -> None: ...

    def _simulate_and_optimize_policy(self): ...

    def _simulate_model(self): ...

    def _optimize_policy(self) -> None: ...
