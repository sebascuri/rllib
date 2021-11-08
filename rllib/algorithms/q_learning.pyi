from torch import Tensor

from rllib.dataset.datatypes import Observation
from rllib.policy import AbstractQFunctionPolicy
from rllib.value_function import AbstractQFunction

from .abstract_algorithm import AbstractAlgorithm

class QLearning(AbstractAlgorithm):
    policy: AbstractQFunctionPolicy
    def compute_optimal_target(
        self, q_function: AbstractQFunction, observation: Observation
    ) -> Tensor: ...

class GradientQLearning(QLearning): ...
