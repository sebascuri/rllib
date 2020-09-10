from typing import Any, Optional, Union

import torch.nn as nn

from rllib.util.parameter_decay import ParameterDecay

from .abstract_algorithm import AbstractAlgorithm
from .kl_loss import KLLoss

class TRPO(AbstractAlgorithm):
    kl_loss: KLLoss
    def __init__(
        self,
        epsilon_mean: Union[ParameterDecay, float] = ...,
        epsilon_var: Optional[Union[ParameterDecay, float]] = ...,
        regularization: bool = ...,
        lambda_: float = ...,
        monte_carlo_target: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
