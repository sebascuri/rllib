"""Model-Based MPPO Agent."""
from typing import Union

from torch.distributions import Distribution
from torch.optim.optimizer import Optimizer

from rllib.algorithms.mppo import MBMPPO
from rllib.dataset.datatypes import Termination
from rllib.model import AbstractModel
from rllib.policy import AbstractPolicy
from rllib.reward import AbstractReward
from rllib.util.parameter_decay import ParameterDecay
from rllib.value_function import AbstractValueFunction

from .model_based_agent import ModelBasedAgent


class MBMPPOAgent(ModelBasedAgent):
    algorithm: MBMPPO

    def __init__(self,
                 model_optimizer: Union[Optimizer, None],
                 policy: AbstractPolicy,
                 value_function: AbstractValueFunction,
                 dynamical_model: AbstractModel,
                 reward_model: AbstractReward,
                 optimizer: Optimizer,
                 mppo_value_learning_criterion,
                 termination: Termination = None,
                 initial_distribution: Distribution = None,
                 plan_horizon: int = 1, plan_samples: int = 8, plan_elites: int = 1,
                 max_memory: int = 10000,
                 model_learn_batch_size: int = 64,
                 model_learn_num_iter: int = 30,
                 bootstrap: bool = True,
                 mppo_epsilon: Union[ParameterDecay, float] = None,
                 mppo_epsilon_mean: Union[ParameterDecay, float] = None,
                 mppo_epsilon_var: Union[ParameterDecay, float] = None,
                 mppo_eta: Union[ParameterDecay, float] = None,
                 mppo_eta_mean: Union[ParameterDecay, float] = None,
                 mppo_eta_var: Union[ParameterDecay, float] = None,
                 mppo_num_action_samples: int = 15,
                 mppo_num_iter: int = 100,
                 mppo_gradient_steps: int = 50,
                 mppo_batch_size: int = None,
                 mppo_target_update_frequency: int = 4,
                 sim_num_steps: int = 200,
                 sim_initial_states_num_trajectories: int = 8,
                 sim_initial_dist_num_trajectories: int = 0,
                 sim_memory_num_trajectories: int = 0,
                 sim_num_subsample: int = 1,
                 sim_max_memory: int = 100000,
                 sim_refresh_interval: int = 1,
                 thompson_sampling: bool = False,
                 train_frequency: int = 0, num_rollouts: int = 1, gamma: float = 1.0,
                 exploration_steps: int = 0, exploration_episodes: int = 0,
                 tensorboard: bool = False,
                 comment: str = '') -> None: ...
