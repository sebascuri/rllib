"""Utilities for GP models."""
from typing import Callable, Optional

from torch import Tensor

from .gps import ExactGP

def add_data_to_gp(
    gp_model: ExactGP, new_inputs: Tensor, new_targets: Tensor
) -> None: ...
def summarize_gp(
    gp_model: ExactGP,
    max_num_points: Optional[int] = ...,
    weight_function: Optional[Callable[[Tensor], Tensor]] = ...,
) -> None: ...
def bkb(gp_model: ExactGP, inducing_points: Tensor, q_bar: float = ...) -> Tensor: ...
