from typing import Any, Optional

import torch.nn as nn
from torch import Tensor

from rllib.dataset.datatypes import Observation

from .abstract_algorithm import AbstractAlgorithm
from .policy_evaluation.abstract_td_target import AbstractTDTarget

class ActorCritic(AbstractAlgorithm):
    standardize_returns: bool
    ope: Optional[AbstractTDTarget]
    def __init__(
        self,
        num_samples: int = ...,
        standardize_returns: bool = ...,
        ope: Optional[AbstractTDTarget] = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def returns(self, trajectory: Observation) -> Tensor: ...
