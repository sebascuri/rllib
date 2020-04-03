"""MPPO Agent Implementation."""

from rllib.agent.abstract_agent import AbstractAgent
from rllib.algorithms.mppo import MPPO
from rllib.dataset.experience_replay import ExperienceReplay

from torch.optim.optimizer import Optimizer


class MPPOAgent(AbstractAgent):
    """Implementation of an agent that runs MPPO."""
    mppo: MPPO
    optimizer: Optimizer
    memory: ExperienceReplay
    target_update_frequency: int
    num_rollouts: int
    num_iter: int
    batch_size: int

    def __init__(self, environment: str, mppo: MPPO, optimizer: Optimizer,
                 memory: ExperienceReplay, num_rollouts: int = 1,
                 num_iter: int = 100, batch_size: int = 64, target_update_frequency: int = 4,
                 gamma: float = 1.0, exploration_steps: int = 0,
                 exploration_episodes: int = 0) -> None: ...
