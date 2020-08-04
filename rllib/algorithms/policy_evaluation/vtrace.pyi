from typing import Any

from torch import Tensor

from .retrace import ReTrace

class VTrace(ReTrace):
    rho_bar: float
    def __init__(self, rho_bar: float = ..., *args: Any, **kwargs: Any) -> None: ...
    def td(
        self, this_v: Tensor, next_v: Tensor, reward: Tensor, correction: Tensor
    ) -> Tensor: ...
