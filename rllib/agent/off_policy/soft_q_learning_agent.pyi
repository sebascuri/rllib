"""Implementation of DQNAgent Algorithms."""
from typing import Union

from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from rllib.agent import QLearningAgent
from rllib.algorithms.q_learning import SoftQLearning
from rllib.dataset import ExperienceReplay
from rllib.policy import SoftMax
from rllib.util.parameter_decay import ParameterDecay
from rllib.value_function import AbstractQFunction


class SoftQLearningAgent(QLearningAgent):
    algorithm: SoftQLearning
    policy: SoftMax

    def __init__(self, q_function: AbstractQFunction, criterion: _Loss,
                 optimizer: Optimizer, memory: ExperienceReplay,
                 temperature: Union[float, ParameterDecay], num_iter: int = 1,
                 batch_size: int = 64, target_update_frequency: int = 4,
                 train_frequency: int = 1, num_rollouts: int = 0, gamma: float = 1.0,
                 exploration_steps: int = 0, exploration_episodes: int = 0,
                 tensorboard: bool = False,
                 comment: str = '') -> None: ...
