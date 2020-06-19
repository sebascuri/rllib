from typing import Tuple

from gym.envs.classic_control.cartpole import CartPoleEnv

from rllib.dataset.datatypes import Action, Done, Reward, State
from rllib.environment.vectorized.util import VectorizedEnv

class VectorizedCartPoleEnv(CartPoleEnv, VectorizedEnv):
    max_torque: float
    discrete: bool
    def __init__(self, discrete: bool = ...) -> None: ...
    def step(self, action: Action) -> Tuple[State, Reward, Done, dict]: ...

class DiscreteVectorizedCartPoleEnv(VectorizedCartPoleEnv):
    def __init__(self) -> None: ...
    def step(self, action: Action) -> Tuple[State, Reward, Done, dict]: ...
