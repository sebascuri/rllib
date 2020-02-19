from rllib.agent import AbstractAgent
from rllib.policy import AbstractPolicy
from rllib.policy.abstract_policy import Distribution
from rllib.dataset import Observation
from rllib.util.gaussian_processes import ExactGPModel
from torch import Tensor
from typing import Iterator


class GPUCBPolicy(AbstractPolicy):
    gp: ExactGPModel
    x: Tensor
    beta: float

    def __init__(self, gp: ExactGPModel, x: Tensor, beta: float = 2.0) -> None: ...

    def __call__(self, state: Tensor) -> Distribution: ...

    def update(self, observation: Observation) -> None: ...

    @property
    def parameters(self) -> Iterator: ...

    @parameters.setter
    def parameters(self, new_params: Iterator) -> None: ...


class GPUCBAgent(AbstractAgent):
    policy: GPUCBPolicy

    def __init__(self, gp: ExactGPModel, x: Tensor, beta: float = 2.0) -> None: ...
