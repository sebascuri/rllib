from ..abstract_agent import AbstractAgent, State, Action, Reward, Done
from rllib.value_function import AbstractQFunction
from rllib.policy import AbstractPolicy
from rllib.exploration_strategies import AbstractExplorationStrategy
from rllib.dataset import ExperienceReplay, Observation
from abc import abstractmethod
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from typing import Tuple


class AbstractDPGAgent(AbstractAgent):
    q_function: AbstractQFunction
    q_target: AbstractQFunction
    policy_target: AbstractPolicy
    exploration: AbstractExplorationStrategy
    memory: ExperienceReplay
    criterion: _Loss
    critic_optimizer: Optimizer
    actor_optimizer: Optimizer
    target_update_frequency: int
    max_action: float
    policy_update_frequency: int
    policy_noise: float
    noise_clip: float

    def __init__(self, q_function: AbstractQFunction, policy: AbstractPolicy,
                 exploration: AbstractExplorationStrategy, criterion: _Loss,
                 critic_optimizer: Optimizer, actor_optimizer: Optimizer,
                 memory: ExperienceReplay, max_action: float = 1.0,
                 target_update_frequency: int = 4, policy_update_frequency: int = 1,
                 policy_noise: float = 0., noise_clip: float = 1.,
                 gamma: float = 1.0,
                 exploration_steps: int = 0, exploration_episodes: int = 0) -> None: ...

    def act(self, state: State) -> Action: ...

    def observe(self, observation: Observation) -> None: ...

    def start_episode(self) -> None: ...

    def end_episode(self) -> None: ...

    def _train(self, batches: int = 1, optimize_actor: bool = True) -> None: ...

    def _train_critic(self, state: State, action: Action, reward: Reward,
                      next_state: State, done: Done, weight: Tensor) -> Tensor: ...

    def _train_actor(self, state: State, weight: Tensor) -> None: ...

    @abstractmethod
    def _td(self, state: State, action: Action, reward: Reward, next_state: State,
            done: Done) -> Tuple[Tensor, Tensor]: ...
