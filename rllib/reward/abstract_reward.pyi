from abc import ABCMeta, abstractmethod
from torch import Tensor
from torch import nn as nn
from rllib.dataset.datatypes import TupleDistribution

class AbstractReward(nn.Module, metaclass=ABCMeta):

    def __init__(self) -> None: ...

    @abstractmethod
    def forward(self, *args: Tensor, **kwargs) -> TupleDistribution: ...
