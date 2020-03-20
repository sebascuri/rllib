from abc import ABCMeta

import torch.nn as nn

class ParameterDecay(nn.Module, metaclass=ABCMeta):
    start: float
    end: float
    decay: float
    step: int

    def __init__(self, start: float, end: float = None, decay: float = None
                 ) -> None: ...

    def update(self) -> None: ...


class Constant(ParameterDecay): ...


class ExponentialDecay(ParameterDecay): ...


class LinearDecay(ParameterDecay): ...