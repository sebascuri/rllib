"""Python Script Template."""
from abc import ABCMeta
from typing import Any

from .abstract_algorithm import AbstractAlgorithm

class DerivedAlgorithm(AbstractAlgorithm, metaclass=ABCMeta):
    base_algorithm: AbstractAlgorithm
    base_algorithm_name: str
    def __init__(
        self, base_algorithm: AbstractAlgorithm, *args: Any, **kwargs: Any
    ) -> None: ...
