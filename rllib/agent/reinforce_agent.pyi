from .abstract_agent import AbstractAgent, State, Action, Reward, Done
from rllib.value_function import AbstractQFunction, AbstractValueFunction
from rllib.policy import AbstractPolicy
from rllib.dataset import Observation
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from typing import Tuple, List


class REINFORCE(AbstractAgent):
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

    def _train(self) -> None: ...

    def _train_actor(self, observations: List[Observation], value_estimates: List[Tensor]) -> None: ...

    def _train_baseline(self, observations: List[Observation], value_estimates: List[Tensor]) -> None: ...

    def _train_critic(self, observations: List[Observation]) -> None: ...

    def _value_estimate(self, trajectories: List[Observation]) -> List[Tensor]: ...

    def _td_base(self, state: State, action: Action, reward: Reward, next_state: State,
                 done: Done, value_estimate: Tensor=None) -> Tuple[Tensor, Tensor]: ...
