from typing import Optional, Union

from torch import Tensor

from rllib.util import ParameterDecay
from rllib.value_function import AbstractQFunction

from ..abstract_policy import AbstractPolicy
from .abstract_q_function_policy import AbstractQFunctionPolicy

class SoftMax(AbstractQFunctionPolicy):
    prior: AbstractPolicy
    def __init__(
        self,
        q_function: AbstractQFunction,
        param: Union[ParameterDecay, float],
        prior: Optional[AbstractPolicy] = None,
    ): ...
    @property
    def temperature(self) -> Tensor: ...
