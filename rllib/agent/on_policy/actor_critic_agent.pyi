from typing import Any, Type

from torch.nn.modules.loss import _Loss

from rllib.algorithms.ac import ActorCritic
from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractQFunction

from .on_policy_agent import OnPolicyAgent

class ActorCriticAgent(OnPolicyAgent):
    """Abstract Implementation of the Policy-Gradient Algorithm.

    The AbstractPolicyGradient algorithm implements the Policy-Gradient algorithm except
    for the computation of the rewards, which leads to different algorithms.

    TODO: build compatible function approximation.

    References
    ----------
    Williams, Ronald J. "Simple statistical gradient-following algorithms for
    connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229-256.
    """

    eps: float = ...
    algorithm: ActorCritic
    def __init__(
        self,
        algorithm_: ActorCritic,
        policy: AbstractPolicy,
        critic: AbstractQFunction,
        criterion: Type[_Loss] = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
