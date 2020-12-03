from typing import Any, List, Optional, Union

from rllib.dataset.datatypes import Loss, Observation
from rllib.model import AbstractModel

from .abstract_algorithm import AbstractAlgorithm
from .abstract_mb_algorithm import AbstractMBAlgorithm

class Dyna(AbstractAlgorithm, AbstractMBAlgorithm):
    only_sim: bool
    def __init__(
        self,
        base_algorithm: AbstractAlgorithm,
        dynamical_model: AbstractModel,
        reward_model: AbstractModel,
        num_steps: int = ...,
        num_samples: int = ...,
        termination_model: Optional[AbstractModel] = ...,
        only_sim: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def forward(
        self, observation: Union[Observation, List[Observation]], **kwargs: Any
    ) -> Loss: ...
