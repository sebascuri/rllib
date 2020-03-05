from .abstract_agent import AbstractAgent
from rllib.policy import AbstractQFunctionPolicy
from rllib.value_function import AbstractQFunction
from rllib.dataset import ExperienceReplay
from rllib.dataset.datatypes import Observation
from rllib.algorithms.q_learning import QLearning
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer


class QLearningAgent(AbstractAgent):
    q_learning: QLearning
    policy: AbstractQFunctionPolicy
    memory: ExperienceReplay
    optimizer: Optimizer
    target_update_frequency: int

    def __init__(self, q_learning:QLearning, q_function: AbstractQFunction,
                 policy: AbstractQFunctionPolicy, criterion: _Loss,
                 optimizer: Optimizer, memory: ExperienceReplay,
                 target_update_frequency: int = 4, gamma: float = 1.0,
                 exploration_steps: int = 0, exploration_episodes: int = 0) -> None: ...

    def observe(self, observation: Observation) -> None: ...

    def start_episode(self) -> None: ...

    def end_episode(self) -> None: ...

    def train(self, batches: int = 1) -> None: ...
