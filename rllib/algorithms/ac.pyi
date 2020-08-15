from typing import Any, Tuple

import torch.nn as nn
from torch import Tensor

from rllib.dataset.datatypes import Observation

from .abstract_algorithm import AbstractAlgorithm

class ActorCritic(AbstractAlgorithm):
    num_samples: int
    entropy_regularization: float
    standardize_returns: bool
    def __init__(
        self,
        num_samples: int = ...,
        entropy_regularization: float = ...,
        standardize_returns: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def returns(self, trajectory: Observation) -> Tensor: ...
    def get_log_p_kl_entropy(
        self, state: Tensor, action: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: ...
