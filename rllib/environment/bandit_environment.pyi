from typing import Tuple

import numpy as np

from rllib.dataset.datatypes import Action, Done, Reward, State
from rllib.reward import AbstractReward

from .abstract_environment import AbstractEnvironment


class BanditEnvironment(AbstractEnvironment):
    reward: AbstractReward
    t: int

    def __init__(self, reward: AbstractReward, num_actions: int = None,
                 x_min: np.ndarray = None, x_max: np.ndarray = None) -> None: ...

    def step(self, action: Action) -> Tuple[State, Reward, Done, dict]: ...

    def reset(self) -> State: ...

    @property
    def state(self) -> State: ...

    @state.setter
    def state(self, value: State) -> None: ...

    @property
    def time(self) -> float: ...
