from typing import Tuple

import numpy as np
from gym.envs.classic_control.pendulum import PendulumEnv

from rllib.dataset.datatypes import Action, Array, Done, Reward, State
from rllib.environment.vectorized.util import VectorizedEnv

class VectorizedPendulumEnv(PendulumEnv, VectorizedEnv):
    def step(self, action: Action) -> Tuple[State, Reward, Done, dict]: ...
    def set_state(self, observation: State) -> None: ...
    def _get_obs(self) -> Array: ...
