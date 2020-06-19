from typing import Optional

from rllib.environment import MDP
from rllib.policy import AbstractPolicy
from rllib.value_function import TabularValueFunction

def linear_system_policy_evaluation(
    policy: AbstractPolicy,
    model: MDP,
    gamma: float,
    value_function: Optional[TabularValueFunction] = ...,
) -> TabularValueFunction: ...
def iterative_policy_evaluation(
    policy: AbstractPolicy,
    model: MDP,
    gamma: float,
    eps: float = ...,
    max_iter: int = ...,
    value_function: Optional[TabularValueFunction] = ...,
) -> TabularValueFunction: ...
