from typing import Type

from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

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
        optimizer: Optimizer,
        criterion: Type[_Loss],
        num_inter: int = ...,
        target_update_frequency: int = ...,
        lambda_: float = ...,
        train_frequency: int = ...,
        num_rollouts: int = ...,
        gamma: float = ...,
        exploration_steps: int = ...,
        exploration_episodes: int = ...,
        tensorboard: bool = ...,
        comment: str = ...,
    ) -> None: ...
