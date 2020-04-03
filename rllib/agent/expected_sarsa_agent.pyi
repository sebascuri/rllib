from typing import List

from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from rllib.algorithms.esarsa import ESARSA
from rllib.dataset.datatypes import Observation
from rllib.policy import AbstractQFunctionPolicy
from rllib.value_function import AbstractQFunction
from .abstract_agent import AbstractAgent


class ExpectedSARSAAgent(AbstractAgent):
    sarsa: ESARSA
    policy: AbstractQFunctionPolicy
    optimizer: Optimizer
    target_update_frequency: int
    num_iter: int
    batch_size: int
    trajectory = List[Observation]

    def __init__(self, environment: str, q_function: AbstractQFunction, policy: AbstractQFunctionPolicy,
                 criterion: _Loss, optimizer: Optimizer, num_inter: int = 1, batch_size: int =1,
                 target_update_frequency: int = 4, gamma: float = 1.0,
                 exploration_steps: int = 0, exploration_episodes: int = 0) -> None: ...
