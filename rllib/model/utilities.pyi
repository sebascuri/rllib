from rllib.dataset.datatypes import State, Action, Reward
from rllib.model import AbstractModel
from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractValueFunction
from typing import Callable

def estimate_value(states: State, model: AbstractModel, policy: AbstractPolicy,
                   reward: Callable[[State, Action], Reward], steps: int,
                   gamma: float = 0.99, bootstrap: AbstractValueFunction = None,
                   num_samples: int = 1) -> Reward: ...