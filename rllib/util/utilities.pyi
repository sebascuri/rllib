from rllib.dataset import Observation
from typing import List
from numpy import ndarray
from torch.distributions import Distribution
from torch import Tensor
import torch


def _mc_value_slow(trajectory: List[Observation], gamma: float = 1.0) -> ndarray: ...

def mc_value(trajectory: List[Observation], gamma: float = 1.0) -> ndarray: ...


def sum_discounted_rewards(trajectory: List[Observation], gamma: float = 1.0
                           ) -> float: ...


class Delta(Distribution):
    arg_constraints: dict
    has_rsample: bool

    def __init__(self, loc: Tensor, validate_args: bool = False) -> None: ...

    @property
    def mean(self) -> Tensor: ...

    @property
    def variance(self) -> Tensor: ...

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> None: ...

    def log_prob(self, value: Tensor) -> float: ...

    def entropy(self) -> float: ...
