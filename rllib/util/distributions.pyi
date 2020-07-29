"""Useful distributions for the library."""

import gpytorch
from torch import Tensor

class Delta(gpytorch.distributions.Delta):
    """Delta Distribution."""

    def __str__(self) -> str: ...
    def entropy(self) -> Tensor: ...
