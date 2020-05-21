"""Off Policy Actor Critic Agent."""

from .off_policy_agent import OffPolicyAgent
from torch.optim.optimizer import Optimizer
from rllib.dataset import ExperienceReplay


class OffPolicyACAgent(OffPolicyAgent):
    """Template for an on-policy algorithm."""

    critic_optimizer: Optimizer
    actor_optimizer: Optimizer
    target_update_frequency: int
    policy_update_frequency: int
    num_iter: int

    def __init__(self, env_name: str,
                 actor_optimizer: Optimizer,
                 critic_optimizer: Optimizer,
                 memory: ExperienceReplay,
                 batch_size: int = 32,
                 train_frequency: int = 1,
                 num_iter: int = 1,
                 target_update_frequency: int = 1,
                 policy_update_frequency: int = 1,
                 gamma: float = 1.0,
                 exploration_steps: int = 0,
                 exploration_episodes: int = 0,
                 comment: str = '') -> None: ...
