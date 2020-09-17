from typing import Any

import torch.nn as nn
from torch import Tensor

from rllib.dataset.datatypes import Observation

from .abstract_algorithm import AbstractAlgorithm

class ActorCritic(AbstractAlgorithm):
    standardize_returns: bool
    def __init__(
        self,
        num_samples: int = ...,
        standardize_returns: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def returns(self, trajectory: Observation) -> Tensor: ...
