from typing import Any, Union

from torch import Tensor

from rllib.dataset.datatypes import Observation
from rllib.util.parameter_decay import ParameterDecay
from rllib.value_function import AbstractValueFunction

from .abstract_algorithm import AbstractAlgorithm, LPLoss

class REPS(AbstractAlgorithm):
    eta: ParameterDecay
    epsilon: Tensor
    value_function: AbstractValueFunction
    def __init__(
        self,
        value_function: AbstractValueFunction,
        epsilon: Union[ParameterDecay, float],
        regularization: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def _policy_weighted_nll(
        self, state: Tensor, action: Tensor, weights: Tensor
    ) -> Tensor: ...
    def _project_eta(self) -> None: ...
    def forward(self, observation: Observation, **kwargs: Any) -> LPLoss: ...
    def update(self): ...
