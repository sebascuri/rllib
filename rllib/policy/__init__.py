"""Implementations of policies that control the environment."""
from .abstract_policy import AbstractPolicy
from .nn_policy import NNPolicy, FelixPolicy
from .tabular_policy import TabularPolicy
from .q_function_policy import AbstractQFunctionPolicy, EpsGreedy, SoftMax, MellowMax
from .random_policy import RandomPolicy
from .mpc_policy import MPCPolicy

"""
Policies map states (and goals) to distributions over actions. 
As (continuous) actions can have many scales, these policies will produce normalized 
actions and then these actions are scaled by the agent the policies before returning the
distribution. 
"""