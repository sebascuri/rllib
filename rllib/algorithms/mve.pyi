from typing import Any, Optional

from rllib.dataset.datatypes import Termination
from rllib.model import AbstractModel
from rllib.reward import AbstractReward

from .abstract_algorithm import AbstractAlgorithm
from .abstract_mb_algorithm import AbstractMBAlgorithm

class MVE(AbstractAlgorithm, AbstractMBAlgorithm):
    td_k: bool
    def __init__(self) -> None: ...

def mve_expand(
    base_algorithm: AbstractAlgorithm,
    dynamical_model: AbstractModel,
    reward_model: AbstractReward,
    num_steps: int = ...,
    num_samples: int = ...,
    termination: Optional[Termination] = ...,
    td_k: bool = ...,
    *args: Any,
    **kwargs: Any,
) -> MVE: ...
