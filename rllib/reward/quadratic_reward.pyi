"""Model for quadratic reward."""

from torch import Tensor

from rllib.dataset.datatypes import TupleDistribution
from rllib.reward import AbstractReward

class QuadraticReward(AbstractReward):
    """Quadratic Reward Function."""

    q: Tensor
    r: Tensor
    def __init__(self, q: Tensor, r: Tensor) -> None: ...
    def forward(self, *args: Tensor, **kwargs) -> TupleDistribution: ...
