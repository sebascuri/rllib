from typing import Any, List, Union

from torch import Tensor

from rllib.dataset.datatypes import Loss, Observation

from .dyna import Dyna

class MVE(Dyna):
    td_k: bool
    lambda_: float
    def __init__(
        self, td_k: bool = ..., lambda_: float = ..., *args: Any, **kwargs: Any,
    ) -> None: ...
    def forward(
        self, observation: Union[Observation, List[Observation]], **kwargs: Any
    ) -> Loss: ...
    def model_augmented_critic_loss(self, observation: Observation) -> Loss: ...
    def get_value_target(self, observation: Observation) -> Tensor: ...
