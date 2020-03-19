from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from rllib.algorithms.q_learning import QLearning
from rllib.dataset import ExperienceReplay
from rllib.policy import AbstractQFunctionPolicy
from rllib.value_function import AbstractQFunction
from .abstract_agent import AbstractAgent


class QLearningAgent(AbstractAgent):
    q_learning: QLearning
    policy: AbstractQFunctionPolicy
    memory: ExperienceReplay
    optimizer: Optimizer
    target_update_frequency: int

    def __init__(self, q_function: AbstractQFunction, policy: AbstractQFunctionPolicy,
                 criterion: _Loss, optimizer: Optimizer, memory: ExperienceReplay,
                 target_update_frequency: int = 4, gamma: float = 1.0,
                 exploration_steps: int = 0, exploration_episodes: int = 0) -> None: ...
