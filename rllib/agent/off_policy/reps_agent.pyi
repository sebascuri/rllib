"""Implementation of REPS Agent stubs."""
from torch.optim.optimizer import Optimizer

from .off_policy_agent import OffPolicyAgent
from rllib.algorithms.reps import REPS
from rllib.dataset import ExperienceReplay


class REPSAgent(OffPolicyAgent):
    algorithm: REPS

    def __init__(self, env_name: str, reps_loss: REPS, optimizer: Optimizer,
                 memory: ExperienceReplay, batch_size: int,
                 num_iter: int,
                 train_frequency: int = 0, num_rollouts: int = 1, gamma: float = 1.0,
                 exploration_steps: int = 0, exploration_episodes: int = 0,
                 comment: str = '') -> None: ...

    def _optimizer_dual(self) -> None: ...

    def _fit_policy(self) -> None: ...

    def _optimize_loss(self, num_iter: int, loss_name: str = 'dual') -> None: ...
