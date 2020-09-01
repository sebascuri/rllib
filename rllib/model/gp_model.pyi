from typing import Any, Callable, Optional, Tuple

import torch.nn as nn
from gpytorch.kernels import Kernel
from gpytorch.means import Mean
from torch import Tensor

from rllib.dataset.datatypes import TupleDistribution

from .abstract_model import AbstractModel

class ExactGPModel(AbstractModel):
    max_num_points: Optional[int]
    input_transform: nn.Module
    likelihood: nn.ModuleList
    gp: nn.ModuleList
    _state: Tensor
    _action: Tensor
    _target: Tensor
    _mean: Optional[Mean] = ...
    _kernel: Optional[Kernel] = ...
    def __init__(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
        mean: Optional[Mean] = ...,
        kernel: Optional[Kernel] = ...,
        input_transform: Optional[nn.Module] = ...,
        max_num_points: Optional[int] = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> TupleDistribution: ...
    def add_data(self, state: Tensor, action: Tensor, next_state: Tensor) -> None: ...
    def summarize_gp(self, weight_function: Optional[nn.Module] = ...) -> None: ...
    def _transform_weight_function(
        self, weight_function: Optional[nn.Module] = ...
    ) -> Callable[[Tensor], Tensor]: ...
    def state_actions_to_input_data(self, state: Tensor, action: Tensor) -> Tensor: ...
    def state_actions_to_train_data(
        self, state: Tensor, action: Tensor, next_state: Tensor
    ) -> Tuple[Tensor, Tensor]: ...

class RandomFeatureGPModel(ExactGPModel):
    """GP Model approximated by Random Fourier Features."""

    approximation: str
    def __init__(
        self, num_features: int, approximation: str = ..., *args: Any, **kwargs: Any
    ) -> None: ...
    def sample_posterior(self) -> None: ...
    def set_prediction_strategy(self, val: str) -> None: ...

class SparseGPModel(ExactGPModel):
    approximation: str
    q_bar: float
    def __init__(
        self,
        inducing_points: Optional[Tensor] = ...,
        q_bar: float = ...,
        approximation: str = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def add_data(self, state: Tensor, action: Tensor, next_state: Tensor) -> None: ...
