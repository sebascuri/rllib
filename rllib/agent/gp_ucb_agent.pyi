from rllib.agent import AbstractAgent
from rllib.policy import AbstractPolicy
from rllib.dataset.datatypes import Observation, Distribution
from rllib.util.gaussian_processes import ExactGP
from torch import Tensor
from typing import Iterator


class GPUCBPolicy(AbstractPolicy):
    gp: ExactGP
    x: Tensor
    beta: float

    def __init__(self, gp: ExactGP, x: Tensor, beta: float = 2.0) -> None: ...

    def forward(self, *args: Tensor, **kwargs) -> Distribution: ...

    def update(self, observation: Observation) -> None: ...


class GPUCBAgent(AbstractAgent):
    policy: GPUCBPolicy

    def __init__(self, gp: ExactGP, x: Tensor, beta: float = 2.0) -> None: ...
