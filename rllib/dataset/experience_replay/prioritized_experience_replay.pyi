from typing import List, Optional, Union

from torch import Tensor

from rllib.dataset.transforms import AbstractTransform
from rllib.util.parameter_decay import ParameterDecay

from .experience_replay import ExperienceReplay

class PrioritizedExperienceReplay(ExperienceReplay):
    alpha: ParameterDecay
    beta: ParameterDecay
    epsilon: Tensor
    max_priority: float
    _priorities: Tensor
    priors: Tensor
    def __init__(
        self,
        max_len: int,
        alpha: Union[float, ParameterDecay] = ...,
        beta: Union[float, ParameterDecay] = ...,
        epsilon: float = ...,
        max_priority: float = ...,
        transformations: Optional[List[AbstractTransform]] = ...,
    ) -> None: ...
    def _update_weights(self) -> None: ...
    @property
    def priorities(self) -> Tensor: ...
    @priorities.setter
    def priorities(self, value: Tensor) -> None: ...
    @property
    def probabilities(self) -> Tensor: ...
