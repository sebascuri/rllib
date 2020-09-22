from typing import Any, Optional, Union

import torch.nn as nn

from rllib.util.parameter_decay import ParameterDecay

from .gaac import GAAC

class TRPO(GAAC):
    monte_carlo_target: bool
    def __init__(
        self,
        epsilon_mean: Union[ParameterDecay, float] = ...,
        epsilon_var: Optional[Union[ParameterDecay, float]] = ...,
        kl_regularization: bool = ...,
        lambda_: float = ...,
        monte_carlo_target: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
