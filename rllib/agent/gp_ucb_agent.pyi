from torch import Tensor

from rllib.agent import AbstractAgent
from rllib.dataset.datatypes import Observation, TupleDistribution
from rllib.policy import AbstractPolicy
from rllib.util.gaussian_processes import ExactGP


class GPUCBPolicy(AbstractPolicy):
    gp: ExactGP
    x: Tensor
    beta: float

    def __init__(self, gp: ExactGP, x: Tensor, beta: float = 2.0) -> None: ...

    def forward(self, *args: Tensor, **kwargs) -> TupleDistribution: ...

    def update(self, observation: Observation) -> None: ...


class GPUCBAgent(AbstractAgent):
    policy: GPUCBPolicy

    def __init__(self, gp: ExactGP, x: Tensor, beta: float = 2.0) -> None: ...
