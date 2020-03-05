from rllib.dataset.datatypes import State
from rllib.model import AbstractModel
from rllib.reward import AbstractReward
from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractValueFunction
from rllib.dataset.datatypes import Observation
from torch import Tensor
from typing import NamedTuple, List


class DynaReturn(NamedTuple):
    q_target: Tensor
    trajectory: List[Observation]


def dyna_rollout(state: State, model: AbstractModel, policy: AbstractPolicy,
                 reward: AbstractReward, steps: int, gamma: float = 0.99,
                 num_samples: int = 1, bootstrap: AbstractValueFunction = None
                 ) -> DynaReturn: ...
