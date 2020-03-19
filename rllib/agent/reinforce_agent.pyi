from typing import List

from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from rllib.dataset.datatypes import Observation
from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractQFunction, AbstractValueFunction
from .abstract_agent import AbstractAgent


class REINFORCEAgent(AbstractAgent):
    trajectories: List[List[Observation]]
    policy_optimizer: Optimizer
    baseline_optimizer: Optimizer
    target_update_frequency: int
    num_rollouts: int

    def __init__(self, policy: AbstractPolicy, policy_optimizer: Optimizer,
                 baseline: AbstractValueFunction = None, critic: AbstractQFunction = None,
                 baseline_optimizer: Optimizer = None, critic_optimizer: Optimizer = None,
                 criterion: _Loss = None, num_rollouts: int = 1, target_update_frequency: int = 1,
                 gamma: float = 1.0, exploration_steps: int = 0, exploration_episodes: int = 0) -> None: ...
