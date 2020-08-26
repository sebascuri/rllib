from typing import Any, Iterable, List

from .abstract_model import AbstractModel

class PredictionStrategy(object):
    models: Iterable[AbstractModel]
    prediction_strategy: str
    _prediction_strategies: List[str]
    _heads: List[int]
    def __init__(
        self, *models: AbstractModel, prediction_strategy: str = ...
    ) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *args: Any) -> None: ...
