from typing import Any, Iterable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from rllib.dataset.datatypes import Action, Done, Reward, State

from .abstract_environment import AbstractEnvironment

try:
    import dm_control.rl.control
    class DMSuiteEnvironment(AbstractEnvironment):
        env: dm_control.rl.control.Environment
        _name: str
        _time: float
        def __init__(
            self,
            env_name: str,
            env_task: str,
            seed: Optional[int] = None,
            **kwargs: Any,
        ) -> None: ...
        @staticmethod
        def _stack_observations(observations: Iterable[np.ndarray]) -> np.ndarray: ...
        def step(self, action: Action) -> Tuple[State, Reward, Done, dict]: ...
        def render(self, mode: str = ...) -> Union[None, np.ndarray, str]: ...
        def close(self) -> None: ...
        def reset(self) -> State: ...
        @property
        def goal(self) -> Union[None, State]: ...
        @property
        def state(self) -> State: ...
        @state.setter
        def state(self, value: State) -> None: ...
        @property
        def time(self) -> float: ...

except Exception:  # dm_control not installed.
    pass
