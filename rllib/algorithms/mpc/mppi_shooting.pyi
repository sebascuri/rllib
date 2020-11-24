from typing import Any, Tuple, Union

from torch import Tensor

from rllib.util.parameter_decay import ParameterDecay

from .abstract_solver import MPCSolver

class MPPIShooting(MPCSolver):
    kappa: ParameterDecay
    filter_coefficients: Tensor
    def __init__(
        self,
        kappa: Union[float, ParameterDecay] = ...,
        filter_coefficients: Tuple[float] = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def get_candidate_action_sequence(self) -> Tensor: ...
    def get_best_action(self, action_sequence: Tensor, returns: Tensor) -> Tensor: ...
    def update_sequence_generation(self, elite_actions: Tensor) -> None: ...
