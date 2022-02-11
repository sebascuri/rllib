from typing import Callable, Optional, Tuple

from torch import Tensor

from rllib.dataset.datatypes import Action, Done, Reward, State
from rllib.model import AbstractModel

from .abstract_environment import AbstractEnvironment

class FakeEnvironment(AbstractEnvironment):
    """A fake environment wraps models into an environment."""

    dynamical_model: AbstractModel
    reward_model: AbstractModel
    termination_model: Optional[AbstractModel]
    initial_state_fn: Callable[..., State]
    _time: int
    _state: Optional[State]
    _name: Optional[str]
    def __init__(
        self,
        dynamical_model: AbstractModel,
        reward_model: AbstractModel,
        initial_state_fn: Callable[..., State],
        termination_model: Optional[AbstractModel] = ...,
        name: Optional[str] = ...,
    ) -> None: ...
    def reset(self) -> State: ...
    def step(self, action: Action) -> Tuple[State, Reward, Done, dict]: ...
    def next_state(
        self, state: Tensor, action: Tensor, next_state: Optional[State] = ...
    ) -> State: ...
    def reward(
        self, state: Tensor, action: Tensor, next_state: Optional[State] = ...
    ) -> Action: ...
    def done(
        self, state: Tensor, action: Tensor, next_state: Optional[State] = ...
    ) -> Done: ...
    def info(self) -> dict: ...
    @property
    def state(self) -> Tensor: ...
    @state.setter
    def state(self, value: Tensor) -> None: ...
    @property
    def time(self) -> int: ...
    @property
    def name(self) -> str: ...
