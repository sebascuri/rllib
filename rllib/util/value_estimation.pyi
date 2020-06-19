from typing import Any, NamedTuple, Optional

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
    rewards: Array, gamma: float = ..., reward_transformer: RewardTransformer = ...
) -> Array: ...
def discount_sum(
    rewards: Tensor, gamma: float = ..., reward_transformer: RewardTransformer = ...,
) -> Array: ...
def mc_return(
    trajectory: Trajectory,
    gamma: float = ...,
    value_function: Optional[AbstractValueFunction] = ...,
    entropy_reg: float = ...,
    reward_transformer: RewardTransformer = ...,
) -> Tensor: ...
def mb_return(
    state: State,
    dynamical_model: AbstractModel,
    reward_model: AbstractReward,
    policy: AbstractPolicy,
    num_steps: int = ...,
    gamma: float = ...,
    num_samples: int = ...,
    value_function: Optional[AbstractValueFunction] = ...,
    entropy_reg: float = ...,
    termination: Optional[Termination] = ...,
    reward_transformer: RewardTransformer = ...,
    **kwargs: Any,
) -> MBValueReturn: ...
