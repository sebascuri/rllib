"""MPPO Agent Implementation."""

from rllib.agent.off_policy_agent import OffPolicyAgent
from rllib.algorithms.mppo import MPPO
from rllib.dataset.experience_replay import ExperienceReplay

from torch.optim.optimizer import Optimizer


class MPPOAgent(OffPolicyAgent):
    """Implementation of an agent that runs MPPO."""
    mppo: MPPO
    optimizer: Optimizer
    target_update_frequency: int
    num_iter: int

    def __init__(self, env_name: str, mppo: MPPO, optimizer: Optimizer,
                 memory: ExperienceReplay,
                 num_iter: int = 100, batch_size: int = 64,
                 target_update_frequency: int = 4,
                 train_frequency: int = 0, num_rollouts: int = 1, gamma: float = 1.0,
                 exploration_steps: int = 0, exploration_episodes: int = 0,
                 comment: str = '') -> None: ...
