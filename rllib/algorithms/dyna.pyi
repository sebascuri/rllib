from typing import Any, List, Optional, Union

from rllib.dataset.datatypes import Loss, Observation, Termination
from rllib.model import AbstractModel
from rllib.reward import AbstractReward

from .abstract_algorithm import AbstractAlgorithm
from .abstract_mb_algorithm import AbstractMBAlgorithm

class Dyna(AbstractAlgorithm, AbstractMBAlgorithm):
    def __init__(self) -> None: ...
    def forward(
        self, observation: Union[Observation, List[Observation]], **kwargs: Any
    ) -> Loss: ...

def dyna_expand(
    base_algorithm: AbstractAlgorithm,
    dynamical_model: AbstractModel,
    reward_model: AbstractReward,
    num_steps: int = ...,
    num_samples: int = ...,
    termination: Optional[Termination] = ...,
    td_k: bool = ...,
    *args: Any,
    **kwargs: Any,
) -> Dyna: ...
