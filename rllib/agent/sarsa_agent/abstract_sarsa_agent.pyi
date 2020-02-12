from ..abstract_agent import AbstractAgent, State, Action, Reward, Done
from abc import abstractmethod
from rllib.policy import AbstractQFunctionPolicy
from rllib.value_function import AbstractQFunction
from rllib.dataset import Observation
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from typing import Tuple, Union


class AbstractSARSAAgent(AbstractAgent):
    q_function: AbstractQFunction
    q_target: AbstractQFunction
    policy: AbstractQFunctionPolicy
    criterion: _Loss
    optimizer: Optimizer
    target_update_frequency: int
    _last_observation: Union[None, Observation]

    def __init__(self, q_function: AbstractQFunction,
                 policy: AbstractQFunctionPolicy, criterion: _Loss,
                 optimizer: Optimizer, target_update_frequency: int = 4,
                 gamma: float = 1.0) -> None: ...

    def act(self, state: State) -> Action: ...

    def observe(self, observation: Observation) -> None: ...

    def start_episode(self) -> None: ...

    def end_episode(self) -> None: ...

    def _train(self, observation: Observation, next_action: Action) -> None: ...

    @abstractmethod
    def _td(self, state: State, action: Action, reward: Reward, next_state: State,
            done: Done, next_action: Action) -> Tuple[Tensor, Tensor]: ...
