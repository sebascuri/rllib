"""Implementations of policies that control the environment."""
from .abstract_policy import AbstractPolicy
from .nn_policy import NNPolicy, FelixPolicy
from .tabular_policy import TabularPolicy
from .q_function_policy import AbstractQFunctionPolicy, EpsGreedy, SoftMax, MellowMax
from .random_policy import RandomPolicy
