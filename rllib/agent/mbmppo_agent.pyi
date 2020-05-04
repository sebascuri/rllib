"""Model-Based MPPO Agent."""
from typing import Union

from torch.distributions import Distribution
from torch.optim.optimizer import Optimizer

from .abstract_model_based_agent import ModelBasedAgent
from rllib.algorithms.mppo import MBMPPO


class MBMPPOAgent(ModelBasedAgent):
    mppo: MBMPPO
    mppo_optimizer: Optimizer
    num_gradient_steps: int

    def __init__(self, env_name: str, mppo: MBMPPO,
                 model_optimizer: Union[Optimizer, None], mppo_optimizer: Optimizer,
                 initial_distribution: Distribution = None,
                 plan_horizon: int = 1, plan_samples: int = 8, plan_elite: int = 1,
                 max_memory: int = 10000, batch_size: int = 64,
                 num_model_iter: int = 30,
                 num_mppo_iter: int = 100,
                 num_simulation_steps: int = 200,
                 num_simulation_trajectories: int = 8,
                 num_gradient_steps: int = 50,
                 num_subsample: int = 1,
                 num_distribution_trajectories: int = 0,
                 num_dataset_trajectories: int = 0,
                 gamma: float = 1.0, exploration_steps: int = 0,
                 exploration_episodes: int = 0,
                 comment: str = '') -> None: ...

    def _optimize_policy(self) -> None: ...
