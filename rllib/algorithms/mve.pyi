from typing import Any, Optional

from rllib.model import AbstractModel

from .abstract_algorithm import AbstractAlgorithm
from .abstract_mb_algorithm import AbstractMBAlgorithm

class MVE(AbstractAlgorithm, AbstractMBAlgorithm):
    td_k: bool
    base_algorithm_name: str
    def __init__(self) -> None: ...

def mve_expand(
    base_algorithm: AbstractAlgorithm,
    dynamical_model: AbstractModel,
    reward_model: AbstractModel,
    num_steps: int = ...,
    num_samples: int = ...,
    termination_model: Optional[AbstractModel] = ...,
    td_k: bool = ...,
    *args: Any,
    **kwargs: Any,
) -> MVE: ...
