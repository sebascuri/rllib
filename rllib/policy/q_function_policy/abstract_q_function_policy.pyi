from abc import ABCMeta
from typing import Union

from torch import Tensor

from rllib.util import ParameterDecay
from rllib.value_function import AbstractQFunction

from ..abstract_policy import AbstractPolicy


class AbstractQFunctionPolicy(AbstractPolicy, metaclass=ABCMeta):
    q_function: AbstractQFunction
    param: ParameterDecay

    def __init__(self, q_function: AbstractQFunction,
                 param: Union[float, ParameterDecay]) -> None: ...

    def forward(self, *args: Tensor, **kwargs) -> Tensor: ...
