from ..abstract_agent import AbstractAgent, State, Action, Reward, Done
from rllib.value_function import AbstractValueFunction
from rllib.policy import AbstractPolicy
from rllib.dataset import Observation
from abc import abstractmethod
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from typing import Tuple, List


class AbstractEpisodicPolicyGradient(AbstractAgent):
    """Abstract Implementation of the Policy-Gradient Algorithm.

    The AbstractPolicyGradient algorithm implements the Policy-Gradient algorithm except
    for the computation of the rewards, which leads to different algorithms.

    TODO: build compatible function approximation.

    References
    ----------
    Williams, Ronald J. "Simple statistical gradient-following algorithms for
    connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229-256.
    """
    trajectories: List[List[Observation]]
    policy: AbstractPolicy
    policy_target: AbstractPolicy
    policy_optimizer: Optimizer
    baseline: AbstractValueFunction
    baseline_optimizer: Optimizer
    critic: AbstractValueFunction
    critic_target: AbstractValueFunction
    critic_optimizer: Optimizer
    criterion: _Loss
    target_update_freq: int
    num_rollouts: int
    eps: float = 1e-12

    def __init__(self, policy: AbstractPolicy, policy_optimizer: Optimizer,
                 baseline: AbstractValueFunction = None, critic: AbstractValueFunction = None,
                 baseline_optimizer: Optimizer = None, critic_optimizer: Optimizer = None,
                 criterion: _Loss = None, num_rollouts: int = 1, target_update_frequency: int = 1,
                 gamma: float = 1.0, exploration_steps: int = 0, exploration_episodes: int = 0) -> None: ...

    def _train(self) -> None: ...

    def _train_actor(self, observations: List[Observation], value_estimates: List[Tensor]) -> None: ...

    def _train_baseline(self, observations: List[Observation], value_estimates: List[Tensor]) -> None: ...

    def _train_critic(self, observations: List[Observation]) -> None: ...

    @abstractmethod
    def _td_base(self, state: State, action: Action, reward: Reward, next_state: State,
                 done: Done, value_estimate: Tensor=None) -> Tuple[Tensor, Tensor]: ...

    @abstractmethod
    def _td_critic(self, state: State, action: Action, reward: Reward, next_state: State,
                 done: Done, ) -> Tuple[Tensor, Tensor]: ...
