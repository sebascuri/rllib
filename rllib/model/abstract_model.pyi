from abc import ABCMeta

import torch.nn as nn
from torch import Tensor

from rllib.dataset.datatypes import TupleDistribution

class AbstractModel(nn.Module, metaclass=ABCMeta):
    dim_state: int
    dim_action: int
    dim_observation: int
    num_states: int
    num_actions: int
    num_observations: int
    discrete_state: bool
    discrete_action: bool
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        dim_observation: int = ...,
        num_states: int = ...,
        num_actions: int = ...,
        num_observations: int = ...,
    ) -> None: ...
    def forward(self, *args: Tensor, **kwargs) -> TupleDistribution: ...
    @property
    def name(self) -> str: ...
    def sample_posterior(self) -> None: ...
    def set_prediction_strategy(self, val: str) -> None: ...
    def scale(self, state: Tensor, action: Tensor) -> Tensor: ...
