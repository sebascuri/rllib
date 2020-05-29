"""MPC Agent Implementation."""

from typing import Optional

from torch.distributions import Distribution
from torch.optim.optimizer import Optimizer

from rllib.algorithms.td import ModelBasedTDLearning
from rllib.policy.mpc_policy import MPCPolicy

from .model_based_agent import ModelBasedAgent


class MPCAgent(ModelBasedAgent):
    """Implementation of an agent that runs an MPC policy."""
    value_optimizer: Optional[Optimizer]
    value_gradient_steps: int
    value_learning: ModelBasedTDLearning

    def __init__(self,
                 env_name: str,
                 mpc_policy: MPCPolicy,
                 model_learn_num_iter: int = 0,
                 model_learn_batch_size: int = 64,
                 model_optimizer: Optimizer = None,
                 value_optimizer: Optimizer = None,
                 max_memory: int = 1000,
                 value_opt_num_iter: int = 0,
                 value_opt_batch_size: int = None,
                 value_num_steps_returns: int = 1,
                 value_gradient_steps: int = 50,
                 sim_num_steps: int = 0,
                 sim_initial_states_num_trajectories: int = 0,
                 sim_initial_dist_num_trajectories: int = 0,
                 sim_memory_num_trajectories: int = 0,
                 initial_distribution: Distribution = None,
                 thompson_sampling: bool = False,
                 train_frequency: int = 0, num_rollouts: int = 1, gamma: float = 1.0,
                 exploration_steps: int = 0, exploration_episodes: int = 0,
                 comment: str = '') -> None: ...
