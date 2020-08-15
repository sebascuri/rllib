from typing import Any, Union

from torch import Tensor

from rllib.util.parameter_decay import ParameterDecay
from rllib.value_function import AbstractValueFunction

from .abstract_algorithm import AbstractAlgorithm

class REPS(AbstractAlgorithm):
    eta: ParameterDecay
    epsilon: Tensor
    critic: AbstractValueFunction
    def __init__(
        self,
        epsilon: Union[ParameterDecay, float],
        regularization: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def _policy_weighted_nll(
        self, state: Tensor, action: Tensor, weights: Tensor
    ) -> Tensor: ...
    def _project_eta(self) -> None: ...
    def update(self): ...
