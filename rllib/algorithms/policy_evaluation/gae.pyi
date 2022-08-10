from typing import Any, Optional, Union

import torch.nn as nn
from torch import Tensor
from rllib.util.utilities import RewardTransformer

from rllib.value_function import AbstractValueFunction

class GAE(nn.Module):
    value_function: Union[AbstractValueFunction, None]
    lambda_gamma: float
    reward_transformer: RewardTransformer
    def __init__(
        self,
        td_lambda: float,
        gamma: float,
        reward_transformer: RewardTransformer = ...,
        value_function: Optional[AbstractValueFunction] = ...,
    ) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> Tensor: ...
