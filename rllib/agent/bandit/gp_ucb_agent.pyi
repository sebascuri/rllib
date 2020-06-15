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
        self, gp: ExactGP, x: Tensor, beta: float = 2.0, noisy: bool = False
    ) -> None: ...
    def forward(self, *args: Tensor, **kwargs) -> TupleDistribution: ...

class GPUCBAgent(AbstractAgent):
    policy: GPUCBPolicy
    def __init__(
        self,
        gp: ExactGP,
        x: Tensor,
        beta: float = 2.0,
        noisy: bool = False,
        tensorboard: bool = False,
    ) -> None: ...
