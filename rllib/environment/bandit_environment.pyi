from .abstract_environment import AbstractEnvironment
from gym import spaces
import numpy as np
from typing import Callable, Tuple
from rllib.dataset.datatypes import State, Action, Reward, Done


class BanditEnvironment(AbstractEnvironment):
    reward: Callable[[Action], Reward]
    t: int

    def __init__(self, reward: Callable[[Action], Reward], num_actions: int = None,
                 x_min: np.ndarray = None, x_max: np.ndarray = None) -> None: ...

    def step(self, action: Action) -> Tuple[State, Reward, Done, dict]: ...

    def reset(self) -> State: ...

    @property
    def state(self) -> State: ...

    @state.setter
    def state(self, value: State) -> None: ...

    @property
    def time(self) -> float: ...
