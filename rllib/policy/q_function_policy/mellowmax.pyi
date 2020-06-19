from torch import Tensor

from .abstract_q_function_policy import AbstractQFunctionPolicy

class MellowMax(AbstractQFunctionPolicy):
    @property
    def omega(self) -> Tensor: ...
