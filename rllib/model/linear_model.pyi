from typing import Any, Optional

from torch import Tensor
from torch.distributions import MultivariateNormal

from rllib.dataset.datatypes import TupleDistribution

from .abstract_model import AbstractModel

class LinearModel(AbstractModel):
    a: Tensor
    b: Tensor
    noise: Optional[MultivariateNormal]
    def __init__(
        self,
        a: Tensor,
        b: Tensor,
        noise: Optional[MultivariateNormal] = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> TupleDistribution: ...
