from typing import NamedTuple, Optional

from torch import Tensor

from rllib.dataset.datatypes import Array, Observation, State
from rllib.model import AbstractModel
from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractValueFunction

from .utilities import RewardTransformer

class MBValueReturn(NamedTuple):
    value_estimate: Tensor
    trajectory: Observation

def reward_to_go(
    rewards: Tensor,
    gamma: float = ...,
    reward_transformer: RewardTransformer = ...,
    terminal_reward: Optional[Tensor] = ...,
) -> Tensor: ...
def discount_cumsum(
    rewards: Array, gamma: float = ..., reward_transformer: RewardTransformer = ...
) -> Array: ...
def discount_sum(
    rewards: Tensor, gamma: float = ..., reward_transformer: RewardTransformer = ...,
) -> Array: ...
def n_step_return(
    observation: Observation,
    gamma: float = ...,
    entropy_regularization: float = ...,
    reward_transformer: RewardTransformer = ...,
    value_function: Optional[AbstractValueFunction] = ...,
    reduction: str = ...,
) -> Tensor: ...
def mc_return(
    observation: Observation,
    gamma: float = ...,
    lambda_: float = ...,
    entropy_regularization: float = ...,
    reward_transformer: RewardTransformer = ...,
    value_function: Optional[AbstractValueFunction] = ...,
    reduction: str = ...,
) -> Tensor: ...
def mb_return(
    state: State,
    dynamical_model: AbstractModel,
    reward_model: AbstractModel,
    policy: AbstractPolicy,
    num_steps: int = ...,
    gamma: float = ...,
    num_samples: int = ...,
    value_function: Optional[AbstractValueFunction] = ...,
    entropy_reg: float = ...,
    termination_model: Optional[AbstractModel] = ...,
    reward_transformer: RewardTransformer = ...,
    reduction: str = ...,
) -> MBValueReturn: ...
