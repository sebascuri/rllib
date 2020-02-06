from rllib.value_function import TabularValueFunction
from rllib.policy import TabularPolicy
from typing import Tuple
from rllib.environment import MDP


def policy_iteration(model: MDP, gamma: float,
                     eps: float = 1e-6, max_iter: int = 1000,
                     value_function: TabularValueFunction = None
                     ) -> Tuple[TabularPolicy, TabularValueFunction]: ...
