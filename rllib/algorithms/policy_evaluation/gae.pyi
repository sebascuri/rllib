from typing import Any, Optional, Union

import torch.nn as nn
from torch import Tensor

from rllib.value_function import AbstractValueFunction

class GAE(nn.Module):
    value_function: Union[AbstractValueFunction, None]
    lambda_gamma: float
    def __init__(
        self,
        lambda_: float,
        gamma: float,
        value_function: Optional[AbstractValueFunction] = None,
    ) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> Tensor: ...
