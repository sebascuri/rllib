from .abstract_agent import AbstractAgent
from rllib.value_function import AbstractQFunction, AbstractValueFunction
from rllib.policy import AbstractPolicy
from rllib.dataset.datatypes import Observation, State, Action, Reward, Done
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from typing import Tuple, List


class REINFORCEAgent(AbstractAgent):
    trajectories: List[List[Observation]]
    policy: AbstractPolicy
    policy_target: AbstractPolicy
    policy_optimizer: Optimizer
    baseline: AbstractValueFunction
    baseline_optimizer: Optimizer
    criterion: _Loss
    target_update_freq: int
    num_rollouts: int
    eps: float = 1e-12

    def __init__(self, policy: AbstractPolicy, policy_optimizer: Optimizer,
                 baseline: AbstractValueFunction = None, critic: AbstractQFunction = None,
                 baseline_optimizer: Optimizer = None, critic_optimizer: Optimizer = None,
                 criterion: _Loss = None, num_rollouts: int = 1, target_update_frequency: int = 1,
                 gamma: float = 1.0, exploration_steps: int = 0, exploration_episodes: int = 0) -> None: ...
