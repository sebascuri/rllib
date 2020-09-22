from typing import Any, Union

import torch.nn as nn

from rllib.util.parameter_decay import ParameterDecay

from .abstract_algorithm import AbstractAlgorithm

class SAC(AbstractAlgorithm):
    def __init__(
        self,
        eta: Union[ParameterDecay, float] = ...,
        entropy_regularization: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
