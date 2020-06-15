from typing import NamedTuple

from torch import Tensor

from rllib.dataset.datatypes import Array, State, Termination, Trajectory
from rllib.model import AbstractModel
from rllib.policy import AbstractPolicy
from rllib.reward import AbstractReward
from rllib.value_function import AbstractValueFunction

from .utilities import RewardTransformer

class MBValueReturn(NamedTuple):
    value_estimate: Tensor
    trajectory: Trajectory

def discount_cumsum(
    rewards: Array,
    gamma: float = 1.0,
    reward_transformer: RewardTransformer = RewardTransformer(),
) -> Array: ...
def discount_sum(
    rewards: Tensor,
    gamma: float = 1.0,
    reward_transformer: RewardTransformer = RewardTransformer(),
) -> Array: ...
def mc_return(
    trajectory: Trajectory,
    gamma: float = 1.0,
    value_function: AbstractValueFunction = None,
    entropy_reg: float = 0.0,
    reward_transformer: RewardTransformer = RewardTransformer(),
) -> Tensor: ...
def mb_return(
    state: State,
    dynamical_model: AbstractModel,
    reward_model: AbstractReward,
    policy: AbstractPolicy,
    num_steps: int = 1,
    gamma: float = 1.0,
    num_samples: int = 1,
    value_function: AbstractValueFunction = None,
    entropy_reg: float = 0.0,
    termination: Termination = None,
    reward_transformer: RewardTransformer = RewardTransformer(),
    **kwargs,
) -> MBValueReturn: ...
