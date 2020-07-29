from abc import ABCMeta, abstractmethod
from typing import Any, Optional

from torch import Tensor
from torch import nn as nn

from rllib.dataset.datatypes import TupleDistribution

class AbstractReward(nn.Module, metaclass=ABCMeta):
    goal: Optional[Tensor]
    def __init__(self, goal: Optional[Tensor] = ...) -> None: ...
    @abstractmethod
    def forward(self, *args: Tensor, **kwargs: Any) -> TupleDistribution: ...
    def set_goal(self, goal: Optional[Tensor]) -> None: ...
