from rllib.algorithms.gaac import GAAC
from .actor_critic_agent import ActorCriticAgent
from rllib.value_function import AbstractValueFunction
from rllib.policy import AbstractPolicy
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer


class GAACAgent(ActorCriticAgent):
    actor_critic: GAAC

    def __init__(self, policy: AbstractPolicy, actor_optimizer: Optimizer,
                 critic: AbstractValueFunction, critic_optimizer: Optimizer,
                 criterion: _Loss, num_rollouts: int = 1, target_update_frequency: int = 1,
                 lambda_: float = 0.97, gamma: float = 1.0, exploration_steps: int = 0, exploration_episodes: int = 0) -> None: ...
