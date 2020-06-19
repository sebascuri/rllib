"""Utilities for the transformers."""

from torch import Tensor

def rescale(tensor: Tensor, scale: Tensor) -> Tensor: ...
def update_mean(
    old_mean: Tensor, old_count: Tensor, new_mean: Tensor, new_count: Tensor
) -> Tensor: ...
def update_var(
    old_mean: Tensor,
    old_var: Tensor,
    old_count: Tensor,
    new_mean: Tensor,
    new_var: Tensor,
    new_count: Tensor,
) -> Tensor: ...
