from .abstract_agent import AbstractAgent, State, Action, Reward, Done
from rllib.value_function import AbstractQFunction
from rllib.policy import AbstractPolicy
from rllib.exploration_strategies import AbstractExplorationStrategy
from rllib.dataset import ExperienceReplay, Observation
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from typing import Tuple


class DDPGAgent(AbstractAgent):
    q_function: AbstractQFunction
    q_target: AbstractQFunction
    _policy: AbstractPolicy
    policy_target: AbstractPolicy
    exploration: AbstractExplorationStrategy
    memory: ExperienceReplay
    criterion: _Loss
    critic_optimizer: Optimizer
    actor_optimizer: Optimizer
    target_update_frequency: int
    max_action: float
    random_steps: int
    policy_update_frequency: int

    def __init__(self, q_function: AbstractQFunction, policy: AbstractPolicy,
                 exploration: AbstractExplorationStrategy, criterion: _Loss,
                 critic_optimizer: Optimizer, actor_optimizer: Optimizer,
                 memory: ExperienceReplay,
                 target_update_frequency: int = 4, gamma: float = 1.0,
                 episode_length: int = None,  max_action: float = 1.0,
                 random_steps: int = 1, policy_update_frequency: int = 1) -> None: ...

    def act(self, state: State) -> Action: ...

    def observe(self, observation: Observation) -> None: ...

    def start_episode(self) -> None: ...

    def end_episode(self) -> None: ...

    def _train(self, batches: int = 1, optimize_actor: bool = True) -> None: ...

    def _train_critic(self, state: State, action: Action, reward: Reward,
                      next_state: State, done: Done, weight: Tensor) -> Tensor: ...

    def _train_actor(self, state: State, weight: Tensor) -> None: ...

    def _td(self, state: State, action: Action, reward: Reward, next_state: State,
            done: Done) -> Tuple[Tensor, Tensor]: ...
