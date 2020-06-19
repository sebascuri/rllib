from typing import Optional, Union

from torch import Tensor

Input = Union[float, Tensor]

def gaussian(x: Input, value_at_1: Input) -> Tensor: ...
def tolerance(
    x: Input, lower: Input, upper: Input, margin: Optional[Input] = ...
) -> Tensor: ...
