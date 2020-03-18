from abc import ABCMeta
from torch import Tensor
from torch.distributions import Categorical

from ..abstract_policy import AbstractPolicy
from rllib.value_function import AbstractQFunction
from rllib.util import ParameterDecay
from typing import Union


class AbstractQFunctionPolicy(AbstractPolicy, metaclass=ABCMeta):
    q_function: AbstractQFunction
    param: ParameterDecay

    def __init__(self, q_function: AbstractQFunction,
                 param: Union[float, ParameterDecay]) -> None: ...

    def forward(self, *args: Tensor, **kwargs) -> Tensor: ...
