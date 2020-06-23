"""Implementations of policies that control the environment.

Policies map states (and goals) to distributions over actions.
As (continuous) actions can have many scales, these policies will produce normalized
actions and then these actions are scaled by the agent the policies before returning the
distribution.

"""
from .abstract_policy import AbstractPolicy
from .derived_policy import DerivedPolicy
from .mpc_policy import MPCPolicy
from .nn_policy import FelixPolicy, NNPolicy
from .q_function_policy import AbstractQFunctionPolicy, EpsGreedy, MellowMax, SoftMax
from .random_policy import RandomPolicy
from .tabular_policy import TabularPolicy
