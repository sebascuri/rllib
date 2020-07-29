from typing import Any

from gpytorch.models import ExactGP
from torch import Tensor

from rllib.dataset.datatypes import TupleDistribution
from rllib.reward import AbstractReward

class GPBanditReward(AbstractReward):
    """A Reward function that is defined through a GP."""

    model: ExactGP
    def __init__(self, model: ExactGP) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> TupleDistribution: ...
