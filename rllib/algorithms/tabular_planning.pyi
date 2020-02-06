from rllib.value_function import TabularValueFunction, NNValueFunction
from rllib.policy import TabularPolicy
from typing import List, Tuple
from rllib.policy import AbstractPolicy
from rllib.environment import MDP


def _init_value_function(num_states: int, terminal_states: List[int]
                         ) -> TabularValueFunction: ...


def inverse_policy_evaluation(policy: AbstractPolicy, model: MDP, gamma: float,
                              value_function: TabularValueFunction = None
                              ) -> TabularValueFunction: ...


def policy_evaluation(policy: AbstractPolicy, model: MDP, gamma: float,
                      eps: float = 1e-6, max_iter: int = 1000,
                      value_function: TabularValueFunction = None
                      ) -> TabularValueFunction: ...


def policy_iteration(model: MDP, gamma: float,
                     eps: float = 1e-6, max_iter: int = 1000,
                     value_function: NNValueFunction = None
                     ) -> Tuple[TabularPolicy, TabularValueFunction]: ...


def value_iteration(model: MDP, gamma: float,
                    eps: float = 1e-6, max_iter: int = 1000,
                    value_function: NNValueFunction = None
                    ) -> Tuple[TabularPolicy, TabularValueFunction]: ...