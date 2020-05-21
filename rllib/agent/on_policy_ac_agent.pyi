"""Off Policy Actor Critic Agent."""

from .on_policy_agent import OnPolicyAgent
from torch.optim.optimizer import Optimizer


class OnPolicyACAgent(OnPolicyAgent):
    """Template for an on-policy algorithm."""

    critic_optimizer: Optimizer
    actor_optimizer: Optimizer
    target_update_frequency: int
    num_iter: int

    def __init__(self, env_name: str, actor_optimizer: Optimizer,
                 critic_optimizer: Optimizer,
                 num_iter: int = 1,
                 target_update_frequency: int = 1,
                 train_frequency: int = 0, num_rollouts: int = 1, gamma: float = 1.0,
                 exploration_steps: int = 0, exploration_episodes: int = 0,
                 comment: str = '') -> None: ...
