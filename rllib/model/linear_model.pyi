from typing import Any, Optional

from torch import Tensor

from rllib.dataset.datatypes import TupleDistribution

from .abstract_model import AbstractModel

class LinearModel(AbstractModel):
    a: Tensor
    b: Tensor
    noise: Tensor
    def __init__(
        self, a: Tensor, b: Tensor, noise: Optional[Tensor] = None
    ) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> TupleDistribution: ...
