from typing import Any, List, Optional, Tuple

import torch
import torch.jit
from torch import Tensor

from rllib.dataset.datatypes import TupleDistribution
from rllib.util.neural_networks import Ensemble

from .nn_model import NNModel

class EnsembleModel(NNModel):
    num_heads: int
    nn: Ensemble
    deterministic: bool
    def __init__(
        self,
        dim_state: Tuple,
        dim_action: Tuple,
        num_heads: int,
        num_states: int = ...,
        num_actions: int = ...,
        initial_scale: float = ...,
        prediction_strategy: str = ...,
        layers: Optional[List[int]] = ...,
        biased_head: bool = ...,
        non_linearity: str = ...,
        input_transform: Optional[List[torch.nn.Module]] = ...,
        deterministic: bool = ...,
    ) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> TupleDistribution: ...
    def sample_posterior(self) -> None: ...
    def scale(self, state: Tensor, action: Tensor) -> Tensor: ...
    @torch.jit.export
    def set_head(self, head_ptr: int) -> None: ...
    @torch.jit.export
    def get_head(self) -> int: ...
    @torch.jit.export
    def set_head_idx(self, head_ptr: Tensor) -> None: ...
    @torch.jit.export
    def get_head_idx(self) -> Tensor: ...
    @torch.jit.export
    def set_prediction_strategy(self, prediction: str) -> None: ...
    @torch.jit.export
    def get_prediction_strategy(self) -> str: ...
