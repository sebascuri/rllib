from typing import Any, Optional

from torch import Tensor

from rllib.dataset.datatypes import TupleDistribution
from rllib.model import AbstractModel

class QuadraticReward(AbstractModel):
    q: Tensor
    r: Tensor
    def __init__(self, q: Tensor, r: Tensor, goal: Optional[Tensor] = ...) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> TupleDistribution: ...
