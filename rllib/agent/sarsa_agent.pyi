from .abstract_agent import AbstractAgent
from rllib.algorithms.sarsa import SARSA
from rllib.dataset.datatypes import Observation
from rllib.policy import AbstractQFunctionPolicy
from rllib.value_function import AbstractQFunction
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from typing import Union, List


class SARSAAgent(AbstractAgent):
    sarsa_algorithm: SARSA
    policy: AbstractQFunctionPolicy
    optimizer: Optimizer
    target_update_frequency: int
    _last_observation: Union[None, Observation]
    _batch_size: int
    _trajectory = List[Observation]

    def __init__(self, sarsa_algorithm: SARSA, q_function: AbstractQFunction,
                 policy: AbstractQFunctionPolicy, criterion: _Loss,
                 optimizer: Optimizer, batch_size: int =1,
                 target_update_frequency: int = 4, gamma: float = 1.0,
                 exploration_steps: int = 0, exploration_episodes: int = 0) -> None: ...

    def train(self, trajectory: List[Observation]) -> None: ...
