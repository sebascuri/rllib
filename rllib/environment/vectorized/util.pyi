from abc import ABCMeta
from typing import Any, Callable, Iterable, List, Tuple, Union

import numpy as np
import torch.__spec__ as torch_mod
from gym import Env

from rllib.dataset.datatypes import Action, Array, Done, Reward, State

class VectorizedEnv(Env, metaclass=ABCMeta):
    """Vectorized implementation of Acrobot."""

    @property
    def bk(self) -> Union[np, torch_mod]: ...
    def atan2(self, sin: Array, cos: Array) -> Array: ...
    def clip(self, val: Array, min_val: float, max_val: float) -> Array: ...
    def cat(self, arrays: Iterable[Array], axis: int = ...) -> Array: ...
    def rand(self, min_val: float, max_val: float) -> Array: ...
    def step(self, action: Action) -> Tuple[State, Reward, Done, dict]: ...

def rk4(
    derivs: Callable[..., Array],
    y0: Array,
    t: Iterable[float],
    *args: Any,
    **kwargs: Any,
) -> Array: ...
