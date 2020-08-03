from typing import Any

from torch import Tensor

from rllib.agent import AbstractAgent
from rllib.dataset.datatypes import TupleDistribution
from rllib.policy import AbstractPolicy
from rllib.util.gaussian_processes import ExactGP
from rllib.util.parameter_decay import ParameterDecay

class GPUCBPolicy(AbstractPolicy):
    gp: ExactGP
    x: Tensor
    beta: ParameterDecay
    noisy: bool
    def __init__(
        self, gp: ExactGP, x: Tensor, beta: float = ..., noisy: bool = ...
    ) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> TupleDistribution: ...

class GPUCBAgent(AbstractAgent):
    policy: GPUCBPolicy
    def __init__(
        self,
        gp: ExactGP,
        x: Tensor,
        beta: float = ...,
        noisy: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
