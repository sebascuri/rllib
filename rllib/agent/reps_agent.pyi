"""Implementation of REPS Agent stubs."""
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer

from rllib.agent.abstract_agent import AbstractAgent
from rllib.algorithms.reps import REPS
from rllib.dataset import ExperienceReplay


class REPSAgent(AbstractAgent):
    reps: REPS
    optimizer: Optimizer
    memory: ExperienceReplay
    num_dual_iter: int
    num_policy_iter: int

    num_rollouts: int
    batch_size: int

    def __init__(self, environment: str, reps_loss: REPS, optimizer: Optimizer,
                 memory: ExperienceReplay, num_rollouts: int, batch_size: int,
                 num_dual_iter: int, num_policy_iter: int = 0,
                 gamma: float = 1.0, exploration_steps: int = 0,
                 exploration_episodes: int = 0, comment: str = '') -> None: ...

    def _optimizer_dual(self) -> None: ...

    def _fit_policy(self) -> None: ...

    def _optimize_loss(self, num_iter: int, loss_name: str = 'dual') -> None: ...
