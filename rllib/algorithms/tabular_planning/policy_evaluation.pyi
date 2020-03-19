from rllib.environment import MDP
from rllib.policy import AbstractPolicy
from rllib.value_function import TabularValueFunction


def linear_system_policy_evaluation(policy: AbstractPolicy, model: MDP, gamma: float,
                                    value_function: TabularValueFunction = None
                                    ) -> TabularValueFunction: ...


def iterative_policy_evaluation(policy: AbstractPolicy, model: MDP, gamma: float,
                                eps: float = 1e-6, max_iter: int = 1000,
                                value_function: TabularValueFunction = None
                                ) -> TabularValueFunction: ...
