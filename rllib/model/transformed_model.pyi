from typing import Any, List, Optional, Union

import torch.nn as nn
from torch import Tensor

from rllib.dataset.datatypes import TupleDistribution
from rllib.dataset.datatypes import Observation
from .abstract_model import AbstractModel

class TransformedModel(AbstractModel):

    base_model: AbstractModel
    transformations: nn.ModuleList
    def __init__(
        self,
        base_model: AbstractModel,
        transformations: Union[List[nn.Module], nn.ModuleList],
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def set_prediction_strategy(self, val: str) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> TupleDistribution: ...
    def scale(self, state: Tensor, action: Tensor) -> Tensor: ...
    def transform_inputs(
        self, state: Tensor, action: Tensor, next_state: Optional[Tensor] = ...
    ) -> Observation: ...
    def back_transform_predictions(
        self, obs: Observation, next_state: Tensor, reward: Tensor, done: Tensor
    ) -> Observation: ...
    def predict(
        self, state: Tensor, action: Tensor, next_state: Optional[Tensor] = ...
    ) -> TupleDistribution: ...
