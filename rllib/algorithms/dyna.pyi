from typing import NamedTuple, List, Callable

from torch import Tensor

from rllib.dataset.datatypes import Observation
from rllib.dataset.datatypes import State
from rllib.model import AbstractModel
from rllib.policy import AbstractPolicy
from rllib.reward import AbstractReward
from rllib.value_function import AbstractValueFunction


class DynaReturn(NamedTuple):
    q_target: Tensor
    trajectory: List[Observation]


def dyna_rollout(state: State, model: AbstractModel, policy: AbstractPolicy,
                 reward: AbstractReward, steps: int, gamma: float = 0.99,
                 num_samples: int = 1, value_function: AbstractValueFunction = None,
                 entropy_reg: float = 0.,
                 termination: Callable[[Tensor, Tensor], Tensor] = None
                 ) -> DynaReturn: ...
