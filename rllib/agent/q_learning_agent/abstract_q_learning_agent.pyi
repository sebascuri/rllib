from ..abstract_agent import AbstractAgent, State, Action, Reward, Done
from abc import abstractmethod
from rllib.policy import AbstractQFunctionPolicy
from rllib.value_function import AbstractQFunction
from rllib.dataset import ExperienceReplay, Observation
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from typing import Tuple


class AbstractQLearningAgent(AbstractAgent):
    q_function: AbstractQFunction
    q_target: AbstractQFunction
    policy: AbstractQFunctionPolicy
    memory: ExperienceReplay
    criterion: _Loss
    optimizer: Optimizer
    target_update_frequency: int

    def __init__(self, q_function: AbstractQFunction,
                 policy: AbstractQFunctionPolicy, criterion: _Loss,
                 optimizer: Optimizer, memory: ExperienceReplay,
                 target_update_frequency: int = 4, gamma: float = 1.0,
                 exploration_steps: int = 0, exploration_episodes: int = 0) -> None: ...

    def observe(self, observation: Observation) -> None: ...

    def start_episode(self) -> None: ...

    def end_episode(self) -> None: ...

    def _train(self, batches: int = 1) -> None: ...

    @abstractmethod
    def _td(self, state: State, action: Action, reward: Reward, next_state: State,
            done: Done) -> Tuple[Tensor, Tensor]: ...
