from .abstract_agent import AbstractAgent
from rllib.value_function import AbstractQFunction
from rllib.policy import AbstractPolicy
from rllib.exploration_strategies import AbstractExplorationStrategy
from rllib.dataset import ExperienceReplay
from rllib.algorithms.dpg import DPG
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer


class DPGAgent(AbstractAgent):
    exploration: AbstractExplorationStrategy
    memory: ExperienceReplay
    dpg_algorithm: DPG
    critic_optimizer: Optimizer
    actor_optimizer: Optimizer
    target_update_frequency: int
    max_action: float
    policy_update_frequency: int

    def __init__(self, q_function: AbstractQFunction, policy: AbstractPolicy,
                 exploration: AbstractExplorationStrategy, criterion: _Loss,
                 critic_optimizer: Optimizer, actor_optimizer: Optimizer,
                 memory: ExperienceReplay, max_action: float = 1.0,
                 target_update_frequency: int = 4, policy_update_frequency: int = 1,
                 policy_noise: float = 0., noise_clip: float = 1.,
                 gamma: float = 1.0,
                 exploration_steps: int = 0, exploration_episodes: int = 0) -> None: ...

    def train(self, batches: int = 1, optimize_actor: bool = True) -> None: ...
