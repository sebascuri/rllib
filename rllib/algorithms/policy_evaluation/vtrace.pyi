from typing import Optional

from torch import Tensor

from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractValueFunction

from .retrace import ReTrace

class VTrace(ReTrace):
    rho_bar: float
    def __init__(
        self,
        critic: AbstractValueFunction,
        policy: Optional[AbstractPolicy] = ...,
        rho_bar: float = ...,
        gamma: float = ...,
        lambda_: float = ...,
    ) -> None: ...
    def td(
        self, this_v: Tensor, next_v: Tensor, reward: Tensor, correction: Tensor
    ) -> Tensor: ...
