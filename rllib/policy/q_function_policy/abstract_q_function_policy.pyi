from abc import ABCMeta
from torch import Tensor
from torch.distributions import Categorical

from ..abstract_policy import AbstractPolicy
from rllib.value_function import AbstractQFunction
from rllib.util import ParameterDecay


class AbstractQFunctionPolicy(AbstractPolicy, metaclass=ABCMeta):
    q_function: AbstractQFunction
    param: ParameterDecay

    def __init__(self, q_function: AbstractQFunction, start: float, end: float = None,
                 decay: float = None) -> None: ...

    def forward(self, *args: Tensor, **kwargs) -> Categorical: ...
