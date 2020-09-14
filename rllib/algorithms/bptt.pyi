from typing import Any

from .abstract_algorithm import AbstractAlgorithm
from .abstract_mb_algorithm import AbstractMBAlgorithm

class BPTT(AbstractAlgorithm, AbstractMBAlgorithm):
    def __init__(self, *args: Any, **kwargs: Any,) -> None: ...
