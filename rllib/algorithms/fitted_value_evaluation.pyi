"""Simplest TD-Learning algorithm."""

from typing import Any

from .abstract_algorithm import AbstractAlgorithm

class FittedValueEvaluationAlgorithm(AbstractAlgorithm):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
