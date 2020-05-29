from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from rllib.algorithms.gaac import GAAC
from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractValueFunction

from .actor_critic_agent import ActorCriticAgent


class GAACAgent(ActorCriticAgent):
    algorithm: GAAC

    def __init__(self, env_name: str, policy: AbstractPolicy,
                 critic: AbstractValueFunction, optimizer: Optimizer,
                 criterion: _Loss, num_inter: int = 1,
                 target_update_frequency: int = 1, lambda_: float = 0.97,
                 train_frequency: int = 0, num_rollouts: int = 1, gamma: float = 1.0,
                 exploration_steps: int = 0, exploration_episodes: int = 0,
                 comment: str = '') -> None: ...
