from typing import Any, Optional, Tuple

import torch.nn as nn
from torch import Tensor

from rllib.dataset.datatypes import Observation

from .abstract_algorithm import AbstractAlgorithm
from .policy_evaluation.abstract_td_target import AbstractTDTarget

class ActorCritic(AbstractAlgorithm):
    num_samples: int
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
    def get_ope_weight(
        self, state: Tensor, action: Tensor, log_prob_action: Tensor
    ) -> Tensor: ...
    def get_log_p_kl_entropy(
        self, state: Tensor, action: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: ...
