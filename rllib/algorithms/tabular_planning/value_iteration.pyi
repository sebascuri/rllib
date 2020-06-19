from typing import Optional, Tuple

from rllib.environment import MDP
from rllib.policy import TabularPolicy
from rllib.value_function import TabularValueFunction

def value_iteration(
    model: MDP,
    gamma: float,
    eps: float = ...,
    max_iter: int = ...,
    value_function: Optional[TabularValueFunction] = ...,
) -> Tuple[TabularPolicy, TabularValueFunction]: ...
