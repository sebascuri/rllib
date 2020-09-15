from typing import Any, Union

import torch.nn as nn
from torch import Tensor

from rllib.util.parameter_decay import ParameterDecay

from .abstract_algorithm import AbstractAlgorithm

class SoftActorCritic(AbstractAlgorithm):
    def __init__(
        self,
        eta: Union[ParameterDecay, float] = ...,
        regularization: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
