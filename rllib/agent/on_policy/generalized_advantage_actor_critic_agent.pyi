from typing import Any, Type

from torch.nn.modules.loss import _Loss

from rllib.algorithms.gaac import GAAC
from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractValueFunction

from .actor_critic_agent import ActorCriticAgent

class GAACAgent(ActorCriticAgent):
    algorithm: GAAC
    def __init__(
        self,
        policy: AbstractPolicy,
        critic: AbstractValueFunction,
        criterion: Type[_Loss],
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
