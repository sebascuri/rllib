from abc import ABCMeta

import torch.nn as nn
from torch import Tensor


class AbstractQFunction(nn.Module, metaclass=ABCMeta):
    dim_action: int
    num_actions: int
    discrete_action: bool
    dim_state: int
    num_states: int
    tau: float
    discrete_state: bool

    def __init__(self, dim_state: int, dim_action: int,
                 num_states: int = None, num_actions: int = None, tau: float = 0.) -> None: ...

    def forward(self, *args:Tensor, **kwargs) -> Tensor: ...

class AbstractValueFunction(AbstractQFunction):

    def __init__(self, dim_state: int, num_states: int = None, tau: float = 0.) -> None: ...