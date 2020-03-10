"""Implementation of DQNAgent Algorithms."""
from rllib.agent import QLearningAgent
from rllib.algorithms.q_learning import SoftQLearning
from rllib.policy import SoftMax
from rllib.value_function import AbstractQFunction
from rllib.dataset import ExperienceReplay
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer


class SoftQLearningAgent(QLearningAgent):
    q_learning: SoftQLearning
    policy: SoftMax

    def __init__(self, q_function: AbstractQFunction, criterion: _Loss,
                 optimizer: Optimizer, memory: ExperienceReplay,
                 target_update_frequency: int = 4, temperature: float = 1.0,
                 gamma: float = 1.0, exploration_steps: int = 0, exploration_episodes: int = 0) -> None: ...