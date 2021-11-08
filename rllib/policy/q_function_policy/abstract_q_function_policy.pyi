from abc import ABCMeta
from typing import Any, Union

from torch import Tensor

from rllib.util.multi_objective_reduction import AbstractMultiObjectiveReduction
from rllib.util.parameter_decay import ParameterDecay
from rllib.value_function import AbstractQFunction

from ..abstract_policy import AbstractPolicy

class AbstractQFunctionPolicy(AbstractPolicy, metaclass=ABCMeta):
    q_function: AbstractQFunction
    param: ParameterDecay
    reduction: AbstractMultiObjectiveReduction
    def __init__(
        self,
        q_function: AbstractQFunction,
        param: Union[float, ParameterDecay],
        reduction: AbstractMultiObjectiveReduction = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> Tensor: ...
