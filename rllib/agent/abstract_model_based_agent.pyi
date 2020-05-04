from abc import ABCMeta, abstractmethod
from typing import Callable


from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.distributions import Distribution

from .abstract_agent import AbstractAgent
from rllib.dataset.datatypes import Observation
from rllib.dataset.experience_replay import BootstrapExperienceReplay, ExperienceReplay

from rllib.model.derived_model import TransformedModel
from rllib.reward import AbstractReward
from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractValueFunction


class ModelBasedAgent(AbstractAgent, metaclass=ABCMeta):
    dynamical_model: TransformedModel
    reward_model: AbstractReward
    termination: Callable
    value_function: AbstractValueFunction

    model_optimizer: Optimizer
    dataset: BootstrapExperienceReplay
    sim_dataset: ExperienceReplay
    sim_trajectory: Observation


    model_learn_num_iter: int
    model_learn_batch_size: int

    plan_horizon: int
    plan_samples: int
    plan_elite: int

    policy_opt_num_iter: int
    policy_opt_batch_size: int

    sim_num_steps: int
    sim_initial_states_num_trajectories: int
    sim_initial_dist_num_trajectories: int
    sim_memory_num_trajectories: int
    sim_refresh_interval: int
    sim_num_subsample: int
    initial_distribution: Distribution
    initial_states: Tensor
    new_episode: bool

    def __init__(self,
                 env_name: str,
                 dynamical_model: TransformedModel,
                 reward_model: AbstractReward,
                 model_optimizer: Optimizer,
                 policy: AbstractPolicy,
                 value_function: AbstractValueFunction = None,
                 termination: Callable = None,
                 plan_horizon: int = 1,
                 plan_samples: int = 1,
                 plan_elite: int = 1,
                 model_learn_num_iter: int = 0,
                 model_learn_batch_size: int = 64,
                 max_memory: int = 10000,
                 policy_opt_num_iter: int = 0,
                 policy_opt_batch_size: int = None,
                 sim_num_steps: int = 20,
                 sim_initial_states_num_trajectories: int = 8,
                 sim_initial_dist_num_trajectories: int = 0,
                 sim_memory_num_trajectories: int = 0,
                 sim_refresh_interval: int = 1,
                 sim_num_subsample: int = 1,
                 initial_distribution: Distribution = None,
                 gamma: float = 1.0, exploration_steps: int = 0,
                 exploration_episodes: int = 0, comment: str = '') -> None: ...

    def _plan(self, state: Tensor) -> None: ...

    def _train(self) -> None: ...

    def _train_model(self) -> None: ...

    def _simulate_and_optimize_policy(self): ...

    def _simulate_model(self): ...

    @abstractmethod
    def _optimize_policy(self) -> None: ...
