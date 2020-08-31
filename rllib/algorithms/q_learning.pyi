from typing import Any

from .abstract_algorithm import AbstractAlgorithm

class QLearning(AbstractAlgorithm):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class GradientQLearning(QLearning): ...
