from typing import Any, List, Optional, Union

from rllib.dataset.datatypes import Loss, Observation
from rllib.value_function import AbstractValueFunction

from .abstract_algorithm import AbstractAlgorithm
from .abstract_mb_algorithm import AbstractMBAlgorithm

class DynaAlgorithm(AbstractAlgorithm, AbstractMBAlgorithm):
    base_algorithm: AbstractAlgorithm
    value_function: Optional[AbstractValueFunction]
    def __init__(
        self, base_algorithm: AbstractAlgorithm, *args: Any, **kwargs: Any
    ) -> None: ...
    def forward(
        self, observation: Union[Observation, List[Observation]], **kwargs: Any
    ) -> Loss: ...
