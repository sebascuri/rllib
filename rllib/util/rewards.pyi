from torch import Tensor
from typing import Union

Input = Union[float, Tensor]

def gaussian(x: Input, value_at_1: Input) -> Tensor: ...

def tolerance(x: Input, lower: Input, upper: Input, margin: Input = None) -> Tensor: ...