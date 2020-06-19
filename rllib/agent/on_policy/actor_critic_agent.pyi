from typing import Type

from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from rllib.algorithms.ac import ActorCritic
from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractQFunction

from .on_policy_ac_agent import OnPolicyACAgent

class ActorCriticAgent(OnPolicyACAgent):
    """Abstract Implementation of the Policy-Gradient Algorithm.

    The AbstractPolicyGradient algorithm implements the Policy-Gradient algorithm except
    for the computation of the rewards, which leads to different algorithms.

    TODO: build compatible function approximation.

    References
    ----------
    Williams, Ronald J. "Simple statistical gradient-following algorithms for
    connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229-256.
    """

    eps: float = 1e-12
    algorithm: ActorCritic
    def __init__(
        self,
        policy: AbstractPolicy,
        critic: AbstractQFunction,
        optimizer: Optimizer,
        criterion: Type[_Loss],
        num_iter: int = 1,
        target_update_frequency: int = 1,
        train_frequency: int = 0,
        num_rollouts: int = 1,
        gamma: float = 1.0,
        exploration_steps: int = 0,
        exploration_episodes: int = 0,
        tensorboard: bool = False,
        comment: str = "",
    ) -> None: ...
