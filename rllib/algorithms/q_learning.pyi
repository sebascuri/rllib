from typing import Any

from rllib.policy import AbstractQFunctionPolicy

from .abstract_algorithm import AbstractAlgorithm

class QLearning(AbstractAlgorithm):
    policy: AbstractQFunctionPolicy

class GradientQLearning(QLearning): ...
