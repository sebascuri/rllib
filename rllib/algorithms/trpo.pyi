from typing import Any, Optional, Union

import torch.nn as nn
from torch import Tensor

from rllib.util.parameter_decay import ParameterDecay

from .abstract_algorithm import AbstractAlgorithm

class TRPO(AbstractAlgorithm):

    epsilon_mean: Tensor
    epsilon_var: Tensor
    eta_mean: ParameterDecay
    eta_var: ParameterDecay
    def __init__(
        self,
        regularization: bool = ...,
        epsilon_mean: Union[ParameterDecay, float] = ...,
        epsilon_var: Optional[Union[ParameterDecay, float]] = ...,
        lambda_: float = ...,
        monte_carlo_target: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
