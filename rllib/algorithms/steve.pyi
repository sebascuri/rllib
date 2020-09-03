from typing import Any, Optional

from torch import Tensor

from rllib.dataset.datatypes import Observation
from rllib.model import AbstractModel

from .abstract_algorithm import AbstractAlgorithm
from .abstract_mb_algorithm import AbstractMBAlgorithm

class STEVE(AbstractAlgorithm, AbstractMBAlgorithm):
    num_models: int
    num_q = int
    base_algorithm_name: str
    def __init__(self) -> None: ...
    def get_value_target(self, observation: Observation) -> Tensor: ...

def steve_expand(
    base_algorithm: AbstractAlgorithm,
    dynamical_model: AbstractModel,
    reward_model: AbstractModel,
    num_steps: int = ...,
    num_samples: int = ...,
    termination_model: Optional[AbstractModel] = ...,
    *args: Any,
    **kwargs: Any,
) -> STEVE: ...
