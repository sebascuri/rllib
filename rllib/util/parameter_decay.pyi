from abc import ABCMeta
from typing import Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor

class ParameterDecay(nn.Module, metaclass=ABCMeta):
    start: nn.Parameter
    end: nn.Parameter
    decay: nn.Parameter
    step: int
    def __init__(
        self,
        start: Union[float, Tensor],
        end: Optional[Union[float, Tensor]] = ...,
        decay: Optional[Union[float, Tensor]] = ...,
    ) -> None: ...
    def update(self) -> None: ...

class Constant(ParameterDecay): ...

class Learnable(ParameterDecay):
    positive: bool
    def __init__(
        self, start: Union[float, Tensor], positive: Optional[bool] = ...
    ) -> None: ...

class ExponentialDecay(ParameterDecay): ...
class PolynomialDecay(ParameterDecay): ...
class LinearDecay(ParameterDecay): ...
class LinearGrowth(ParameterDecay): ...

class OUNoise(ParameterDecay):
    mean: Tensor
    std_dev: Tensor
    theta: float
    dt: float
    def __init__(
        self,
        mean: Union[float, Tensor] = ...,
        std_deviation: Union[float, Tensor] = ...,
        theta: float = ...,
        dt: float = ...,
        dim: Tuple[int] = ...,
    ) -> None: ...
