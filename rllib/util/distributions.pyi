"""Useful distributions for the library."""

from torch import Tensor
import gpytorch

class Delta(gpytorch.distributions.Delta):  # type: ignore
    """Delta Distribution."""

    def __str__(self) -> str: ...
    def entropy(self) -> Tensor: ...
